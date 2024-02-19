import cv2
from Decoders import decode_function, BeamDecoder
from LPRNet import load_default_lprnet
from SpatialTransformer import load_default_stn
import numpy as np
import torch


def convert_output_image(lpr_output: torch.Tensor) -> np.ndarray:
    converted_lpr_output = lpr_output.squeeze(0).cpu()
    converted_lpr_output = converted_lpr_output.detach().numpy().transpose((1, 2, 0))
    converted_lpr_output = converted_lpr_output.astype('float32')
    converted_lpr_output = 127.5 + converted_lpr_output * 128.
    converted_lpr_output = converted_lpr_output.astype('uint8')

    return converted_lpr_output


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LPRNet_model = load_default_lprnet(device)
    STNet_model = load_default_stn(device)

    license_plate_image_cropped = cv2.imread('../../recognitor_dataset/plates/0.bmp')
    license_plate_image_cropped = cv2.resize(license_plate_image_cropped, (94, 24), interpolation=cv2.INTER_CUBIC)
    license_plate_image_cropped = (np.transpose(np.float32(license_plate_image_cropped), (2, 0, 1)) - 127.5) * 0.0078125

    data = torch.from_numpy(license_plate_image_cropped).float().unsqueeze(0).to(device)

    license_plate_image_transformed = STNet_model(data)
    predictions = LPRNet_model(license_plate_image_transformed)
    predictions = predictions.cpu().detach().numpy()

    labels, probability, predicted_labels = decode_function(
        predictions,
        [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
            'Y', 'X', '-',
        ],
        BeamDecoder
    )

    if (probability[0] < -85) and (len(labels[0]) in [8, 9]):
        print('License plate number {} with {} probability.'.format(labels[0], probability[0]))
