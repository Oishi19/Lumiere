from QTool import QTool

class QToolAnimeGAN(QTool):

    net = None

    def __init__(self, parent=None, toolInput=None, onCompleted=None):
        super(QToolAnimeGAN, self).__init__(parent, "Anime GAN", 
                                              "Transform photos of real-world scenes into anime style images",
                                              "images/anime_2.png", self.onRun, toolInput, onCompleted, self)

        self.parent = parent
        self.output = None

    def onRun(self, progressSignal, args):
        image = args[0]

        from torchvision.transforms.functional import to_tensor, to_pil_image
        from AnimeGANModel import Generator as AnimeGanGenerator
        import torch
        import cv2
        import numpy as np
        from PIL import Image

        # Clean up CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        progressSignal.emit(10, "Checking CUDA capability")
        useGpu = torch.cuda.is_available()
        device = "cuda" if useGpu else "cpu"

        i = 0
        max_attempts = 2 # once on CUDA, once on CPU

        while i < max_attempts:
            try:
                progressSignal.emit(20, "Loading model")

                if QToolAnimeGAN.net == None:
                    net = AnimeGanGenerator()
                    net.load_state_dict(torch.load("models/face_paint_512_v2.pt", map_location=device))
                    net.to(device).eval()

                progressSignal.emit(30, "Loading current pixmap")

                progressSignal.emit(40, "Preprocessing image")

                b, g, r, _ = cv2.split(np.asarray(image))
                image_np = np.dstack((b, g, r))
                image_pil = Image.fromarray(image_np)

                with torch.no_grad():
                    progressSignal.emit(50, "Converting to tensor")
                    image_tensor = to_tensor(image_pil).unsqueeze(0) * 2 - 1

                    progressSignal.emit(60, "Running the model on " + device)

                    out = net(image_tensor.to(device), False # <-- upsample_align (Align corners in decoder upsampling layers)
                              ).cpu()
                    out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5

                    progressSignal.emit(70, "Postprocessing output")

                    out = to_pil_image(out)

                    # Add alpha channel back
                    alpha = np.full((out.height, out.width), 255)
                    out_np = np.dstack((np.asarray(out), alpha)).astype(np.uint8)

                    del image_tensor
                    del net
                    del out

                    i += 1

                    self.output = Image.fromarray(out_np)
                    break

            except RuntimeError as e:
                i += 1
                print(e)
                if device == "cuda":
                    # Retry on CPU
                    progressSignal.emit(10, "Failed to run on CUDA device. Retrying on CPU")
                    device = "cpu"
                    torch.cuda.empty_cache()
                    print("Retrying on CPU")