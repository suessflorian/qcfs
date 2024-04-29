import argparse
from models import modelpool
from preprocess import datapool
from funcs import *
from checkpoint import load
from conversion import primed, convert_snn
import foolbox as fb
from tqdm import tqdm
from utils import *
from foolbox.attacks import LinfFastGradientAttack, LinfDeepFoolAttack, LInfFMNAttack

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vgg16')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--id', default=None, type=str, help='model identifier')
    parser.add_argument('--device', default='mps', type=str, help='mps or cpu')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--epsilon', default=0.01, type=float, help='epsilon')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--attack', type=str, default='fgsm')
    args = parser.parse_args()

    seed_all()

    attack = {
        "fgsm": LinfFastGradientAttack(),
        "deepfool": LinfDeepFoolAttack(),
        "fmn": LInfFMNAttack(),
    }

    if args.attack not in attack:
        raise ValueError(f"unsupported attack '{args.attack}'... supported attacks are: {list(attack.keys())}")
    selected_attack = attack[args.attack]

    # preparing data
    _, test_loader = datapool(args.data, 16)
    # preparing model
    model = modelpool(args.model, args.data)
    model = primed(model, args.t)
    model = load(model, args.id)

    # conversion
    if args.mode == 'snn':
        model = convert_snn(model)

    model.to(args.device)
    reset_net(model)


    success_perturbed_images = []
    success_perturbed_predictions = []
    success_original_predictions = []
    success_original_images = []
    success_original_labels = []
    n_total = 0
    n_total_correct = 0

    model.eval()
    for images, labels in tqdm(test_loader, desc='Attack Batches', unit="Batch"):
        images, labels = images.to(args.device), labels.to(args.device)

        fmodel = fb.PyTorchModel(model, bounds=(-3, 3), device=args.device) # TODO: should just train on 0-1s
        raw_attack, perturbed_image, _ = selected_attack(fmodel, images, labels, epsilons=args.epsilon)
        with torch.no_grad():
            reset_net(model)
            original_predictions = model(images).argmax(dim = -1)
            reset_net(model)
            perturbed_predictions = model(perturbed_image).argmax(dim = -1)

        correct_pre_attack = (original_predictions == labels)
        correct_post_attack = (perturbed_predictions == labels)

        successful_attack_indices = (correct_pre_attack & ~correct_post_attack).view(-1)
        n_successful_attacks = successful_attack_indices.sum().item()

        n_total_correct += correct_pre_attack.sum().item()

        if n_successful_attacks == 0:
            continue

        success_perturbed_images.append(perturbed_image[successful_attack_indices])
        success_perturbed_predictions.append(perturbed_predictions[successful_attack_indices])
        success_original_predictions.append(original_predictions[successful_attack_indices])
        success_original_images.append(images[successful_attack_indices])
        success_original_labels.append(labels[successful_attack_indices])

        n_total += len(labels)

    perturbed_images = torch.cat(success_perturbed_images, dim=0)
    perturbed_predictions = torch.cat(success_perturbed_predictions, dim=0)
    original_predictions = torch.cat(success_original_predictions, dim=0)
    original_images = torch.cat(success_original_images, dim=0)
    original_labels = torch.cat(success_original_labels, dim=0)

    print(f"\nNumber of Successful Attacks: {perturbed_images.shape[0]}")
    print(f"Number of Total Images Examined: {n_total}")
    if n_total > 0:
        print(f"Attack Success Rate: {perturbed_images.shape[0] / n_total_correct*100:.2f}%\n")

    ############################## Plotting ##############################
    # for index in range(len(original_images)):
    #     plot_attack(original_images.cpu(),
    #                 perturbed_images.cpu(),
    #                 original_labels.cpu(),
    #                 original_predictions.cpu(),
    #                 perturbed_predictions.cpu(),
    #                 index=index,
    #                 dataset=model_name.split('-')[0].lower())

    #     if index < len(original_images) - 1 and input('Print More? [y, n]: ') == 'n':
    #         break


