from tools.dataset import create_dataset
from tools.utils import setup_seed


if __name__ == '__main__':
    setup_seed(20)
    create_dataset()
