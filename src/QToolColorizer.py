from QTool import QTool
import os

class QToolColorizer(QTool):
    def __init__(self, parent=None, toolInput=None, onCompleted=None):
        super(QToolColorizer, self).__init__(parent, "Interactive Colorization", 
                                             "Colourize image interactively by leveraging a vision transformer",
                                             "images/colourizer.png", 
                                             self.onRun, toolInput, onCompleted, self)
        self.parent = parent
        self.output = None

    def onRun(self, progressSignal):
        import ColorizerMain
        import ColorizerModeling
        from timm.models import create_model
        import torch
        import os
        from FileUtils import merge_files

        # Merge NN model files into pth file if not exists
        if not os.path.exists("models/icolorit_base_4ch_patch16_224.pth"):
            merge_files("icolorit_base_4ch_patch16_224.pth", "models")

        def get_model():
            model = create_model(
                "icolorit_base_4ch_patch16_224",
                pretrained=False,
                pretrained_cfg=None,
                drop_path_rate=0.0,
                drop_block_rate=None,
                use_rpb=True,
                avg_hint=True,
                head_mode="cnn",
                mask_cent=False,
            )

            return model

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = get_model()
        model.to(device)
        checkpoint = torch.load(os.path.join("models", "icolorit_base_4ch_patch16_224.pth"), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        self.output = model