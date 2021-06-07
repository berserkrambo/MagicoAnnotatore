from path import Path
import numpy as np
import click


def create_test_train_arturo_half(ds_path):
    anomaly_folder = Path(ds_path) / 'anomaly'
    normal_folder = Path(ds_path) / 'normal'
    train_folder = Path(ds_path) / 'arturo_half' / 'train'
    train_folder.makedirs_p()
    test_folder = Path(ds_path) / 'arturo_half' / 'test'
    test_folder.makedirs_p()

    anomaly_files = [f for f in sorted(anomaly_folder.files('*.jpg')) if f.find('_anomaly_') >= 0]
    normal_files = [f for f in sorted(normal_folder.files('*.jpg')) if f.find('_normal_') >= 0]
    np.random.shuffle(anomaly_files)
    np.random.shuffle(normal_files)

    # print(f"total normal images: {len(normal_files)}")
    print(f"total anomaly images: {len(anomaly_files)}")

    test_len = int(len(anomaly_files) * 0.1)
    print(f"copying {test_len} normal images to test folder")
    for i in normal_files[:test_len]:
        i.copy(test_folder)
    print(f"copying {test_len} anomaly images to test folder")
    for i in anomaly_files[:test_len]:
        i.copy(test_folder)

    print(f"copying {len(anomaly_files) - test_len} normal images to train folder")
    for i in normal_files[test_len:len(anomaly_files)]:
        i.copy(train_folder)
    print(f"copying {len(anomaly_files) - test_len} anomaly images to train folder")
    for i in anomaly_files[test_len:]:
        i.copy(train_folder)


def create_test_train_arturo_base(ds_path):
    anomaly_folder = Path(ds_path) / 'anomaly'
    normal_folder = Path(ds_path) / 'normal'
    train_folder = Path(ds_path) / 'arturo' / 'train'
    train_folder.makedirs_p()
    test_folder = Path(ds_path) / 'arturo' / 'test'
    test_folder.makedirs_p()

    anomaly_files = [f for f in sorted(anomaly_folder.files('*.jpg')) if f.find('_anomaly_') >= 0]
    normal_files = [f for f in sorted(normal_folder.files('*.jpg')) if f.find('_normal_') >= 0]
    np.random.shuffle(anomaly_files)
    np.random.shuffle(normal_files)

    print(f"total normal images: {len(normal_files)}")
    print(f"total anomaly images: {len(anomaly_files)}")

    test_len = int(len(normal_files) * 0.15)
    print(f"copying {test_len} normal images to test folder")
    for i in normal_files[:test_len]:
        i.copy(test_folder)
    print(f"copying {len(normal_files) - test_len} normal images to train folder")
    for i in normal_files[test_len:]:
        i.copy(train_folder)

    test_len = int(len(anomaly_files) * 0.2)
    print(f"copying {test_len} anomaly images to test folder")
    for i in anomaly_files[:test_len]:
        i.copy(test_folder)


def all_reduce(src_path, dst_path):
    src_path, dst_path = Path(src_path), Path(dst_path)
    anomaly_folder = dst_path / 'anomaly'
    anomaly_folder.makedirs_p()
    normal_folder = dst_path / 'normal'
    normal_folder.makedirs_p()

    for dir in src_path.dirs():
        an_dir = dir / "anomaly"
        no_dir = dir / "normal"
        if an_dir.isdir() and no_dir.isdir():
            for f in an_dir.files('*.jpg'):
                f.copy(anomaly_folder)
            for f in no_dir.files('*.jpg'):
                f.copy(normal_folder)
        else:
            print(f"error processing dir {dir.name}, 'anomaly' and 'normal' folder not found inside")



## va indicata la dir contenente le cartelle normal e anomaly ottenute eseguento anonano_divider.py
## come secondo argomento la cartella dove riversare tutte le anomaly e normal appena create con anonano_divider.py
@click.command()
@click.option('--source_path', type=str, required=True)
@click.option('--dest_path', type=str, required=True)
def main(source_path, dest_path):
    all_reduce(source_path, dest_path)
    create_test_train_arturo_base(dest_path)
    create_test_train_arturo_half(dest_path)


if __name__ == '__main__':
    main()
