from tkinter import Tk, Canvas, Button, PhotoImage, filedialog, Label, Toplevel
from PIL import Image, ImageTk
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / \
    Path(r"C:\Users\Acer Aspire 3\Downloads\Test\GUI Folder\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def stylize_image(content_img_path, style_img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imsize = 512 if torch.cuda.is_available() else 128

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    def image_loader(image_path):
        image = Image.open(image_path)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)

    assert style_img.size() == content_img.size(
    ), "we need to import style and content images of the same size"

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
            self.std = torch.tensor(std).view(-1, 1, 1).to(device)

        def forward(self, img):
            return (img - self.mean) / self.std

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        normalization = Normalization(normalization_mean, normalization_std)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(
                    layer.__class__.__name__))
            model.add_module(name, layer)
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        return model, style_losses, content_losses

    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        model, style_losses, content_losses = get_style_model_and_losses(
            cnn, normalization_mean, normalization_std, style_img, content_img)
        input_img = input_img.requires_grad_(True)
        model.eval().to(device)
        optimizer = get_input_optimizer(input_img)
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                return style_score + content_score
            optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean,
                                cnn_normalization_std, content_img, style_img, content_img.clone())
    out_t = (output.data.squeeze())
    output_img = transforms.ToPILImage()(out_t)
    return output_img


def select_content_image(content_label):
    global content_image_path
    content_image_path = filedialog.askopenfilename()
    content_img = Image.open(content_image_path)
    content_img = content_img.resize((256, 256), Image.LANCZOS)
    content_img = ImageTk.PhotoImage(content_img)
    content_label.config(image=content_img)
    content_label.image = content_img
    return content_image_path


def select_style_image(style_label):
    global style_image_path
    style_image_path = filedialog.askopenfilename()
    style_img = Image.open(style_image_path)
    style_img = style_img.resize((256, 256), Image.LANCZOS)
    style_img = ImageTk.PhotoImage(style_img)
    style_label.config(image=style_img)
    style_label.image = style_img
    return style_image_path


def display_result_window(result_img):
    result_window = Toplevel(window)
    result_window.title("Result Image")
    result_window.geometry("1200x1200")
    result_label = Label(result_window)
    result_label.pack()
    result_img = ImageTk.PhotoImage(result_img)
    result_label.config(image=result_img)
    result_label.image = result_img


window = Tk()
window.geometry("620x607")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=607,
    width=620,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    620.0,
    607.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    15.0,
    16.0,
    604.0,
    595.0,
    fill="#D2E7CC",
    outline="")

canvas.create_text(
    147.0,
    47.0,
    anchor="nw",
    text="Photo Style App",
    fill="#000000",
    font=("Italiana Regular", 64 * -1)
)

canvas.create_rectangle(
    20.0,
    112.0,
    599.0,
    117.0,
    fill="#000000",
    outline="")

canvas.create_text(
    63.0,
    148.0,
    anchor="nw",
    text="Select Content Image",
    fill="#000000",
    font=("Italiana Regular", 36 * -1)
)

canvas.create_text(
    387.0,
    148.0,
    anchor="nw",
    text="Select Style Image",
    fill="#000000",
    font=("Italiana Regular", 36 * -1)
)

content_label = Label(window)
content_label.place(x=50, y=200, width=256, height=256)

style_label = Label(window)
style_label.place(x=350, y=200, width=256, height=256)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_content_image(content_label),
    relief="flat"
)
button_1.place(
    x=49.0,
    y=424.0,
    width=211.0,
    height=40.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_style_image(style_label),
    relief="flat"
)
button_2.place(
    x=359.0,
    y=424.0,
    width=211.0,
    height=40.0
)


def apply_style_transfer():
    if content_image_path and style_image_path:
        output_img = stylize_image(content_image_path, style_image_path)
        display_result_window(output_img)


button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=apply_style_transfer,
    relief="flat"
)
button_3.place(
    x=169.0,
    y=500.0,
    width=282.0,
    height=50.0
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    156.0,
    280.0,
    image=image_image_1
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    464.0,
    280.0,
    image=image_image_2
)

window.resizable(False, False)
window.mainloop()
