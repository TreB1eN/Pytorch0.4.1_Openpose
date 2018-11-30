from openpose import Openpose
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train openpose")
    parser.add_argument("-r", "--resume", help="whether resume from the latest saved model",action="store_true")
    parser.add_argument("-save", "--from_save_folder", help="whether resume from the save path",action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    openpose = Openpose()
    if args.resume:
        openpose.resume_training_load(from_save_folder = args.from_save_folder)
    openpose.train()