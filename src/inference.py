class ModelInference:
    """Handles model inference for new images."""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = ImageProcessor()
        
        try:
            # Load model with better error handling
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_type = checkpoint.get('model_type', 'ResNet18')
            
            self.model = MycorrhizalCNN(model_type=model_type, num_classes=5)
            
            # Check if state dict is compatible
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except KeyError:
                raise ValueError("Invalid model checkpoint - missing model_state_dict")
            except RuntimeError as e:
                raise ValueError(f"Model architecture mismatch: {e}")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        # Class mappings
        self.class_names = [
            "Not colonized",
            "Lightly colonized", 
            "Moderately colonized",
            "Heavily colonized",
            "Not annotated"
        ]
        
        # Colonization percentages for each class (approximate)
        self.class_percentages = {
            0: 0,     # Not colonized
            1: 25,    # Lightly colonized
            2: 50,    # Moderately colonized
            3: 80,    # Heavily colonized
            4: 0      # Not annotated
        }
