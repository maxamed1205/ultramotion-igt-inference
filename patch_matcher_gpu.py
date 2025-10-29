# Patch pour ajouter les méthodes GPU-resident manquantes à matcher.py

import re

def add_gpu_methods_to_matcher():
    """Ajoute les méthodes GPU-resident manquantes au HungarianMatcher"""
    
    file_path = r"c:\Users\maxam\Desktop\TM\ultramotion-igt-inference\src\core\inference\d_fine\matcher.py"
    
    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Méthodes à ajouter avant la fin de la classe
    gpu_methods = '''
    def _convert_indices_for_topk_gpu(self, indices):
        """Convert GPU indices to format compatible with topk without CPU transfer"""
        # Return indices as-is for GPU processing - no CPU conversion needed
        return indices

    def _ensure_gpu_indices(self, indices):
        """Ensure indices are in proper GPU tensor format"""
        if isinstance(indices, list):
            return [(torch.as_tensor(i, dtype=torch.int64, device='cuda'), 
                    torch.as_tensor(j, dtype=torch.int64, device='cuda')) 
                   for i, j in indices]
        return indices

    def _gpu_topk_assignment(self, C, sizes):
        """GPU-based topk assignment approximation"""
        indices = []
        offset = 0
        for batch_idx, size in enumerate(sizes):
            batch_cost = C[batch_idx, :, offset:offset + size]
            target_to_query = torch.argmin(batch_cost, dim=0)
            
            query_indices = []
            target_indices = []
            used_queries = set()
            
            for target_idx in range(size):
                best_query = int(target_to_query[target_idx])
                if best_query not in used_queries:
                    query_indices.append(best_query)
                    target_indices.append(target_idx)
                    used_queries.add(best_query)
                    
            indices.append((
                torch.tensor(query_indices, dtype=torch.int64, device=C.device),
                torch.tensor(target_indices, dtype=torch.int64, device=C.device)
            ))
            offset += size
        return indices
'''
    
    # Trouver le dernier endroit dans la classe (avant la fin)
    # Chercher la dernière méthode de la classe
    class_end_pattern = r'(\n\s*return indices_list\n.*?(?=\n\n|\Z))'
    
    # Ajouter les méthodes juste avant la fin de la classe
    if 'def _convert_indices_for_topk_gpu' not in content:
        # Trouver un bon endroit pour insérer (après get_top_k_matches)
        insert_pos = content.rfind('        return indices_list')
        if insert_pos != -1:
            # Trouver la fin de cette méthode
            next_line_pos = content.find('\n', insert_pos)
            content = content[:next_line_pos] + gpu_methods + content[next_line_pos:]
    
    # Écrire le fichier modifié
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("✅ Méthodes GPU-resident ajoutées au HungarianMatcher")

if __name__ == "__main__":
    add_gpu_methods_to_matcher()