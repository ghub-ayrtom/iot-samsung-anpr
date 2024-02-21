from Decoders import BeamDecoder, GreedyDecoder
from typing import Sequence
import torch
import torch.nn as nn


class SmallBasicBlock(nn.Module):
    """
        Аргументы:
            in_channels - количество каналов во входной карте объектов
            out_channels - количество каналов на выходе базового блока
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(SmallBasicBlock, self).__init__()
        intermediate_channels = out_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    """
        Аргументы:
            class_num - количество всех возможных символов (классов)
            dropout_prob - вероятность обнуления нейрона в слое nn.Dropout()
            out_indices - индексы слоёв, из которых мы хотим извлечь карты объектов и использовать их для встраивания в
            глобальный контекст
    """

    def __init__(self,
                 class_num: int,
                 dropout_prob: float,
                 out_indices: Sequence[int]):
        super(LPRNet, self).__init__()

        self.class_num = class_num
        self.out_indices = out_indices

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # -> extract feature map (2)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),

            SmallBasicBlock(in_channels=64, out_channels=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # -> extract feature map (6)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),

            SmallBasicBlock(in_channels=64, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            SmallBasicBlock(in_channels=256, out_channels=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # -> extract feature map (13)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_prob),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=self.class_num),
            nn.ReLU(),  # -> extract feature map (22)
        )

        # in_channels - сумма всех каналов в извлечённых картах объектов (отметки выше)
        self.container = nn.Conv2d(
            in_channels=64 + 128 + 256 + self.class_num,
            out_channels=self.class_num,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

    def forward(self, x):
        extracted_feature_maps = list()

        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)

            if i in self.out_indices:
                extracted_feature_maps.append(x)

        global_contex_emb = list()

        for i, feature_map in enumerate(extracted_feature_maps):
            if i in (0, 1):
                feature_map = nn.AvgPool2d(kernel_size=5, stride=5)(feature_map)
            if i == 2:
                feature_map = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(feature_map)

            f_pow = torch.pow(feature_map, 2)
            f_mean = torch.mean(f_pow)
            feature_map = torch.div(feature_map, f_mean)
            global_contex_emb.append(feature_map)

        x = torch.cat(global_contex_emb, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


# Загружает предобученные веса модели
def load_weights(model, weights, device):
    checkpoint = torch.load(weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint['net_state_dict'])


def load_default_lprnet(device):
    model = LPRNet(
        class_num=23,
        dropout_prob=0,
        out_indices=(2, 6, 13, 22)
    ).to(device)

    load_weights(
        model=model,
        weights='/home/ayrtom/PycharmProjects/iot-samsung-anpr/models/LPRNet/LPRNet_pretrained.ckpt',
        device=device
    )

    model.eval()
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_default_lprnet(device)
    input_ = torch.Tensor(2, 3, 24, 94).to(device)
    output = model(input_)

    print(model, end='\n\n')
    print('Output shape is:', output.shape, end='\n\n')

    beam_decoder = BeamDecoder()
    greedy_decoder = GreedyDecoder()
    predictions = output.cpu().detach().numpy()

    print(beam_decoder.decode(
        predictions,
        [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
            'Y', 'X', '-',
        ]
    ))

    print(greedy_decoder.decode(
        predictions,
        [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
            'Y', 'X', '-',
        ]
    ))
