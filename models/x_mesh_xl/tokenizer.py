import torch
from torch import nn, Tensor
from typing import Tuple
from einops import rearrange, repeat, reduce



def discretize(
    t: Tensor,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    
    lo, hi = continuous_range
    assert hi > lo
    t = (t - lo) / (hi - lo)    # cube normalize
    t *= num_discrete
    t -= 0.5
    return t.round().long().clamp(min = 0, max = num_discrete - 1)



def undiscretize(
    t: Tensor,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo
    t = t.float()
    t += 0.5
    t /= num_discrete       # cube normalize
    return t * (hi - lo) + lo



class MeshTokenizer(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.pad_id = -1
        self.num_discrete_coors = args.n_discrete_size  # default: 800
        self.codebook_size = args.n_discrete_size + 1   # default: 128 + 1 (增加1用于separate token)
        self.separate_token_id = args.n_discrete_size   # separate token
        self.coor_continuous_range = (-1., 1.)
    
    
    def tokenize(self, data_dict: dict) -> dict:
        '''
        Turn 3D meshes into sequential tokens: <bos> [<x>, <y>, <z>], ... <eos>
        with separate token between faces: <bos> [<x>, <y>, <z>] <separate> [<x>, <y>, <z>] ... <eos>
        '''
        
        ### 3D mesh face parsing
        vertices = data_dict['vertices']    # batch x nv x 3
        faces = data_dict['faces']          # batch x nf x 3
        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')   # batch x nf
        
        batch, num_vertices, num_coors = vertices.shape
        _, num_faces, _ = faces.shape

        # fill padding tokens with 0, to prevent gather idx error
        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)
        
        # collect vertice coordinates per-face: b x nf x nv x c
        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)
        face_coords = vertices.gather(-2, faces_vertices.long())
        
        # continuous to discrete face coords: b x nf x nv x c
        discrete_face_coords = discretize(
            face_coords,
            continuous_range=self.coor_continuous_range,
            num_discrete=self.num_discrete_coors
        )
        
        # pad invalid faces with <pad_id>: batch x nf x nv x c
        discrete_padded_coords = discrete_face_coords.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            self.pad_id
        )
        
        
        ### mesh to sequence convertion with separate tokens between faces: batch x ntokens
        # Reshape to get each face as a separate entity: batch x nf x (nv*c)
        face_tokens = discrete_padded_coords.reshape(batch, num_faces, -1)
        
        # Create separate tokens for each face
        separate_tokens = torch.ones(batch, num_faces-1, 1).to(face_tokens.device) * self.separate_token_id
        
        # Initialize list to store tokens for each batch
        batch_tokens = []
        batch_masks = []
        
        # Process each batch separately to insert separate tokens between faces
        for b in range(batch):
            # Get valid faces for this batch
            valid_faces = face_tokens[b][face_mask[b]]
            
            if len(valid_faces) > 0:
                # Insert separate tokens between faces
                tokens_with_separators = []
                for i, face in enumerate(valid_faces):
                    tokens_with_separators.append(face)
                    # Add separator after each face except the last one
                    if i < len(valid_faces) - 1:
                        tokens_with_separators.append(torch.tensor([self.separate_token_id]).to(face.device))
                
                # Concatenate all tokens for this batch
                batch_tokens.append(torch.cat(tokens_with_separators, dim=0))
                # Create attention mask (1 for all tokens)
                batch_masks.append(torch.ones_like(batch_tokens[-1]).float())
            else:
                # Handle empty case
                batch_tokens.append(torch.tensor([]).to(face_tokens.device))
                batch_masks.append(torch.tensor([]).to(face_tokens.device))
        
        # Pad sequences to the same length
        max_len = max([len(tokens) for tokens in batch_tokens])
        input_ids = torch.ones(batch, max_len).to(face_tokens.device) * self.pad_id
        attention_mask = torch.zeros(batch, max_len).to(face_tokens.device)
        
        for b in range(batch):
            if len(batch_tokens[b]) > 0:
                input_ids[b, :len(batch_tokens[b])] = batch_tokens[b]
                attention_mask[b, :len(batch_masks[b])] = batch_masks[b]
        
        # reserve two spots for <bos> and <eos>:
        #     input_ids: <bos> ... <eos> <pad> ... => <pad> ... <pad> <pad> ...
        #     attn_mask:    1  ...    1     0  ... =>    1  ...    1     0  ...
        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        input_ids = torch.cat((place_holder * self.pad_id, input_ids, place_holder * self.pad_id), dim=1)
        attention_mask = torch.cat((place_holder, place_holder, attention_mask), dim=1)
        
        ### meshXL inputs
        data_dict['input_ids'] = input_ids.long()               # batch x (nf * 3 * 3 + 2)
        data_dict['attention_mask'] = attention_mask.float()    # batch x (nf * 3 * 3 + 2)
        
        # discard <bos> and <eos> tokens
        data_dict['codes'] = discrete_padded_coords.long()      # batch x (nf * 3 * 3)
        data_dict['discrete_face_coords'] = discrete_face_coords
        
        return data_dict
    
    
    def detokenize(self, input_ids: Tensor) -> dict:
        '''
        Turn sequential tokens: <bos> [<x>, <y>, <z>] <separate> [<x>, <y>, <z>] ... <eos> into 3D meshes
        '''
        # input_ids: b (n q) or b n q, without <bos> or <eos>
        input_ids = input_ids.reshape(input_ids.shape[0], -1)
        
        # Remove separate tokens and reshape to get faces
        batch_size = input_ids.shape[0]
        output_faces = []
        face_masks = []
        
        for b in range(batch_size):
            # Get tokens for this batch
            tokens = input_ids[b]
            
            # Find indices of separate tokens
            separate_indices = (tokens == self.separate_token_id).nonzero(as_tuple=True)[0]
            
            # If no separate tokens found, process as a single face
            if len(separate_indices) == 0:
                # Check if we have valid tokens (not all padding)
                if torch.any(tokens != self.pad_id):
                    # Reshape to face coordinates (9 tokens per face: 3 vertices x 3 coordinates)
                    face_tokens = tokens[:9]  # Take first 9 tokens as a face
                    output_faces.append(face_tokens.reshape(1, 9))  # 1 face with 9 coordinates
                    face_masks.append(torch.ones(1).to(tokens.device))  # 1 valid face
                else:
                    # All padding, no valid faces
                    output_faces.append(torch.ones(0, 9).to(tokens.device) * self.pad_id)
                    face_masks.append(torch.zeros(0).to(tokens.device))
            else:
                # Process multiple faces separated by separate tokens
                face_start_indices = [0] + (separate_indices + 1).tolist()
                face_end_indices = separate_indices.tolist() + [len(tokens)]
                
                # Extract faces between separators
                faces = []
                masks = []
                
                for start, end in zip(face_start_indices, face_end_indices):
                    # Skip if we don't have enough tokens for a face (9 coordinates)
                    if end - start < 9:
                        continue
                    
                    # Take 9 tokens for each face
                    face = tokens[start:start+9]
                    
                    # Check if face has valid tokens
                    if torch.all(face != self.pad_id):
                        faces.append(face)
                        masks.append(1)
                
                if faces:
                    output_faces.append(torch.stack(faces))
                    face_masks.append(torch.tensor(masks).to(tokens.device))
                else:
                    output_faces.append(torch.ones(0, 9).to(tokens.device) * self.pad_id)
                    face_masks.append(torch.zeros(0).to(tokens.device))
        
        # Pad to same number of faces
        max_faces = max([faces.shape[0] for faces in output_faces]) if output_faces else 0
        
        if max_faces > 0:
            # Create padded tensor for faces
            padded_faces = torch.ones(batch_size, max_faces, 9).to(input_ids.device) * self.pad_id
            padded_masks = torch.zeros(batch_size, max_faces).to(input_ids.device)
            
            # Fill with actual faces
            for b in range(batch_size):
                if output_faces[b].shape[0] > 0:
                    padded_faces[b, :output_faces[b].shape[0]] = output_faces[b]
                    padded_masks[b, :face_masks[b].shape[0]] = face_masks[b]
            
            # Reshape to batch x nface x 3 x 3
            pred_face_coords = rearrange(padded_faces, 'b nf (v c) -> b nf v c', v=3)
            face_mask = padded_masks.bool()
        else:
            # Handle case with no valid faces
            pred_face_coords = torch.ones(batch_size, 0, 3, 3).to(input_ids.device)
            face_mask = torch.zeros(batch_size, 0).bool().to(input_ids.device)
        
        # Back to continuous space
        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete=self.num_discrete_coors,
            continuous_range=self.coor_continuous_range
        )
        
        # Mask padding coordinates out with nan
        continuous_coors = continuous_coors.masked_fill(
            ~rearrange(face_mask, 'b nf -> b nf 1 1'), 
            float('nan')
        )
        
        output_dict = {}
        output_dict['recon_faces'] = continuous_coors
        
        return output_dict
    
    
    def forward(self, data_dict: dict) -> dict:
        
        encoder_output = self.tokenize(data_dict)
        decoder_output = self.detokenize(
            input_ids = encoder_output['codes'], 
        )
        data_dict.update(encoder_output)
        data_dict.update(decoder_output)
        return data_dict
