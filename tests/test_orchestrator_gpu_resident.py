"""
Tests pour valider la refactorisation orchestrator.py en mode GPU-resident
"""
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


def test_prepare_inference_gpu_path():
    """Test que prepare_inference_inputs() en mode GPU-resident fonctionne"""
    
    try:
        from src.core.inference.engine.orchestrator import prepare_inference_inputs
        
        # Mock du frame tensor GPU
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        frame_t = Mock()
        frame_t.tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=device)
        
        # Mock des modèles
        mock_dfine = Mock()
        mock_sam = Mock()
        
        # Mock des fonctions
        with patch('src.core.inference.engine.orchestrator.run_detection') as mock_detection, \
             patch('src.core.inference.engine.orchestrator.run_segmentation') as mock_segmentation, \
             patch('src.core.inference.engine.orchestrator.compute_mask_weights') as mock_weights:
            
            # Configuration des mocks
            mock_detection.return_value = ([10, 10, 100, 100], 0.95)
            mock_segmentation.return_value = torch.ones(256, 256, device=device)
            mock_weights.return_value = {"W_edge": 0.1, "W_in": 0.8, "W_out": 0.1}
            
            # Test avec sam_as_numpy=False (mode GPU-resident)
            result = prepare_inference_inputs(frame_t, mock_dfine, mock_sam, sam_as_numpy=False)
            
            # Vérifications
            assert result["state_hint"] in ("VISIBLE", "LOST")
            
            # Vérifier que run_segmentation a été appelé avec as_numpy=False
            mock_segmentation.assert_called_once()
            call_args = mock_segmentation.call_args
            assert 'as_numpy' in call_args.kwargs
            assert call_args.kwargs['as_numpy'] == False
            
            print("✅ Test GPU-resident path passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import orchestrator: {e}")


def test_prepare_inference_cpu_path():
    """Test que prepare_inference_inputs() en mode legacy CPU fonctionne"""
    
    try:
        from src.core.inference.engine.orchestrator import prepare_inference_inputs
        
        # Mock du frame tensor
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        frame_t = Mock()
        frame_t.tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=device)
        
        # Mock des modèles
        mock_dfine = Mock()
        mock_sam = Mock()
        
        # Mock des fonctions
        with patch('src.core.inference.engine.orchestrator.run_detection') as mock_detection, \
             patch('src.core.inference.engine.orchestrator.run_segmentation') as mock_segmentation, \
             patch('src.core.inference.engine.orchestrator.compute_mask_weights') as mock_weights:
            
            # Configuration des mocks
            mock_detection.return_value = ([10, 10, 100, 100], 0.95)
            mock_segmentation.return_value = np.ones((256, 256), dtype=np.uint8)
            mock_weights.return_value = {"W_edge": 0.1, "W_in": 0.8, "W_out": 0.1}
            
            # Test avec sam_as_numpy=True (mode legacy)
            result = prepare_inference_inputs(frame_t, mock_dfine, mock_sam, sam_as_numpy=True)
            
            # Vérifications
            assert result["state_hint"] in ("VISIBLE", "LOST")
            
            # Vérifier que run_segmentation a été appelé avec as_numpy=True
            mock_segmentation.assert_called_once()
            call_args = mock_segmentation.call_args
            assert 'as_numpy' in call_args.kwargs
            assert call_args.kwargs['as_numpy'] == True
            
            print("✅ Test legacy CPU path passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import orchestrator: {e}")


def test_prepare_inference_default_behavior():
    """Test que le comportement par défaut (sam_as_numpy=False) utilise le mode GPU-resident"""
    
    try:
        from src.core.inference.engine.orchestrator import prepare_inference_inputs
        
        # Mock du frame tensor
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        frame_t = Mock()
        frame_t.tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=device)
        
        # Mock des modèles
        mock_dfine = Mock()
        mock_sam = Mock()
        
        # Mock des fonctions
        with patch('src.core.inference.engine.orchestrator.run_detection') as mock_detection, \
             patch('src.core.inference.engine.orchestrator.run_segmentation') as mock_segmentation, \
             patch('src.core.inference.engine.orchestrator.compute_mask_weights') as mock_weights:
            
            # Configuration des mocks
            mock_detection.return_value = ([10, 10, 100, 100], 0.95)
            mock_segmentation.return_value = torch.ones(256, 256, device=device)
            mock_weights.return_value = {"W_edge": 0.1, "W_in": 0.8, "W_out": 0.1}
            
            # Test sans spécifier sam_as_numpy (devrait utiliser False par défaut)
            result = prepare_inference_inputs(frame_t, mock_dfine, mock_sam)
            
            # Vérifications
            assert result["state_hint"] in ("VISIBLE", "LOST")
            
            # Vérifier que run_segmentation a été appelé avec as_numpy=False par défaut
            mock_segmentation.assert_called_once()
            call_args = mock_segmentation.call_args
            assert 'as_numpy' in call_args.kwargs
            assert call_args.kwargs['as_numpy'] == False  # Comportement par défaut
            
            print("✅ Test default behavior (GPU-resident) passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import orchestrator: {e}")


@patch('src.core.inference.engine.orchestrator.time')
def test_kpi_instrumentation_orchestrator(mock_time):
    """Test que l'instrumentation KPI fonctionne dans l'orchestrator"""
    
    mock_time.time.return_value = 1234567890.0
    
    try:
        with patch('core.monitoring.kpi.safe_log_kpi') as mock_log_kpi, \
             patch('core.monitoring.kpi.format_kpi') as mock_format_kpi:
            
            mock_format_kpi.return_value = "formatted_kpi"
            
            from src.core.inference.engine.orchestrator import prepare_inference_inputs
            
            # Mock du frame tensor
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            frame_t = Mock()
            frame_t.tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=device)
            
            # Mock des modèles et fonctions
            mock_dfine = Mock()
            mock_sam = Mock()
            
            with patch('src.core.inference.engine.orchestrator.run_detection') as mock_detection, \
                 patch('src.core.inference.engine.orchestrator.run_segmentation') as mock_segmentation, \
                 patch('src.core.inference.engine.orchestrator.compute_mask_weights') as mock_weights:
                
                mock_detection.return_value = ([10, 10, 100, 100], 0.95)
                mock_segmentation.return_value = torch.ones(256, 256, device=device)
                mock_weights.return_value = {"W_edge": 0.1, "W_in": 0.8, "W_out": 0.1}
                
                # Test avec sam_as_numpy=False
                prepare_inference_inputs(frame_t, mock_dfine, mock_sam, sam_as_numpy=False)
                
                # Vérifier que les fonctions KPI ont été appelées
                mock_format_kpi.assert_called_once()
                mock_log_kpi.assert_called_once_with("formatted_kpi")
                
                # Vérifier le contenu du KPI
                call_args = mock_format_kpi.call_args[0][0]
                assert call_args["event"] == "prepare_inference_inputs"
                assert call_args["sam_as_numpy"] == 0  # False -> 0
                assert call_args["ts"] == 1234567890.0
                
                print("✅ Test KPI instrumentation in orchestrator passed")
        
    except ImportError as e:
        pytest.skip(f"Cannot import orchestrator or KPI modules: {e}")


def test_gpu_tensor_processing():
    """Test spécifique pour le traitement des tensors GPU"""
    
    try:
        from src.core.inference.engine.orchestrator import prepare_inference_inputs
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU tensor test")
        
        # Créer un tensor CUDA réel
        device = torch.device("cuda:0")
        frame_t = Mock()
        frame_t.tensor = torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8, device=device)
        
        mock_dfine = Mock()
        mock_sam = Mock()
        
        with patch('src.core.inference.engine.orchestrator.run_detection') as mock_detection, \
             patch('src.core.inference.engine.orchestrator.run_segmentation') as mock_segmentation, \
             patch('src.core.inference.engine.orchestrator.compute_mask_weights') as mock_weights:
            
            # Capturer l'image passée à run_segmentation
            captured_image = None
            def capture_segmentation_call(*args, **kwargs):
                nonlocal captured_image
                captured_image = args[1]  # full_image argument
                return torch.ones(256, 256, device=device)
            
            mock_detection.return_value = ([10, 10, 100, 100], 0.95)
            mock_segmentation.side_effect = capture_segmentation_call
            mock_weights.return_value = {"W_edge": 0.1, "W_in": 0.8, "W_out": 0.1}
            
            # Test GPU-resident
            result = prepare_inference_inputs(frame_t, mock_dfine, mock_sam, sam_as_numpy=False)
            
            # Vérifier que l'image est restée sur GPU
            assert captured_image is not None
            assert hasattr(captured_image, 'device')
            assert captured_image.device == device
            assert captured_image.dtype == torch.float32
            
            print(f"✅ GPU tensor processing test passed - device: {captured_image.device}")
        
    except ImportError as e:
        pytest.skip(f"Cannot import orchestrator: {e}")


if __name__ == "__main__":
    print("🧪 Running orchestrator GPU-resident tests...")
    
    test_prepare_inference_gpu_path()
    test_prepare_inference_cpu_path()
    test_prepare_inference_default_behavior()
    test_kpi_instrumentation_orchestrator(Mock())
    test_gpu_tensor_processing()
    
    print("✅ All orchestrator tests completed!")