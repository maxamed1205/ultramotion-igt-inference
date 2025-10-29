# Patch pour remplacer get_top_k_matches dans matcher.py par la version GPU-resident

import re

def replace_get_top_k_matches():
    """Remplace get_top_k_matches par la version GPU-resident complète"""
    
    file_path = r"c:\Users\maxam\Desktop\TM\ultramotion-igt-inference\src\core\inference\d_fine\matcher.py"
    
    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Nouvelle méthode GPU-resident complète
    new_method = '''    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # Use GPU or CPU path based on self.use_gpu_match
        if self.use_gpu_match:
            # GPU-resident topk - avoid CPU conversion
            C_work = C.clone()  # Work on GPU copy
            for i in range(k):
                if i == 0 and initial_indices is not None:
                    # Use initial indices for first iteration
                    indices_k = initial_indices if isinstance(initial_indices, list) else [initial_indices]
                else:
                    # GPU-based assignment approximation
                    indices_k = self._gpu_topk_assignment(C_work, sizes)
                    
                indices_list.append(indices_k)
                
                # Mask used assignments for next iteration on GPU
                for batch_idx, (query_idx, target_idx) in enumerate(indices_k):
                    if batch_idx < len(sizes):
                        offset = sum(sizes[:batch_idx])
                        # Mask the cost matrix entries to prevent reuse
                        if hasattr(query_idx, '__iter__'):
                            for q, t in zip(query_idx, target_idx):
                                C_work[batch_idx, q, offset + t] = 1e6
                        else:
                            C_work[batch_idx, query_idx, offset + target_idx] = 1e6
        else:
            # Legacy CPU path
            C_cpu = C.cpu() if hasattr(C, 'cpu') else C
            for i in range(k):
                indices_k = (
                    [linear_sum_assignment(c[i]) for i, c in enumerate(C_cpu.split(sizes, -1))]
                    if i > 0
                    else initial_indices
                )
                indices_list.append(
                    [
                        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                        for i, j in indices_k
                    ]
                )
                for c, idx_k in zip(C.split(sizes, -1), indices_k):
                    idx_k = np.stack(idx_k)
                    c[:, idx_k] = 1e6
                    
        # Concatenate results for each batch
        final_indices = []
        for j in range(len(sizes)):
            query_list = []
            target_list = []
            for i in range(min(k, len(indices_list))):
                if j < len(indices_list[i]):
                    q_idx, t_idx = indices_list[i][j]
                    if hasattr(q_idx, '__iter__') and not isinstance(q_idx, torch.Tensor):
                        query_list.extend(q_idx)
                        target_list.extend(t_idx)
                    else:
                        query_list.append(q_idx)
                        target_list.append(t_idx)
            
            final_indices.append((
                torch.cat([torch.as_tensor([q], dtype=torch.int64) for q in query_list]) if query_list else torch.tensor([], dtype=torch.int64),
                torch.cat([torch.as_tensor([t], dtype=torch.int64) for t in target_list]) if target_list else torch.tensor([], dtype=torch.int64)
            ))
        
        return final_indices'''
    
    # Trouver et remplacer la méthode get_top_k_matches
    pattern = r'def get_top_k_matches\(self, C, sizes, k=1, initial_indices=None\):.*?return indices_list'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_method.strip(), content, flags=re.DOTALL)
        print("✅ Méthode get_top_k_matches remplacée par version GPU-resident")
    else:
        print("❌ Impossible de trouver la méthode get_top_k_matches")
        return
    
    # Écrire le fichier modifié
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    replace_get_top_k_matches()