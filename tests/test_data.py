import csv
from pathlib import Path
import pytest
from zara_bag_retrieval.data import read_inventory, scan_dataset_role, assert_inventory_matches


def test_windows_absolute_paths_resolve_to_local_dataset(tmp_path):
    root = tmp_path / 'zara_retrieval_dataset'
    file = root / 'gallery' / 'img 001' / 'front.jpg'
    file.parent.mkdir(parents=True)
    file.write_bytes(b'not a real jpeg needed for path-only test')
    inventory = tmp_path / 'gallery_inventory.csv'
    with inventory.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['product_id','path'])
        writer.writeheader()
        writer.writerow({'product_id':'img 001', 'path':r'C:\Users\example\Pictures\zara_retrieval_dataset\gallery\img 001\front.jpg'})
    actual = scan_dataset_role(root, 'gallery')
    cached = read_inventory(inventory, root, 'gallery')
    assert_inventory_matches(actual, cached, 'gallery')
    assert cached[0].path == file.resolve()


def test_inventory_mismatch_fails_fast(tmp_path):
    root = tmp_path / 'dataset'
    one = root / 'gallery' / 'bag_a' / 'one.jpg'
    one.parent.mkdir(parents=True)
    one.write_bytes(b'one')
    actual = scan_dataset_role(root, 'gallery')
    other = list(actual)
    other[0] = type(actual[0])('bag_b', actual[0].path, 'bag_b/one.jpg')
    with pytest.raises(ValueError, match='changed'):
        assert_inventory_matches(actual, other, 'gallery')


def test_prevent_parent_path_traversal(tmp_path):
    root = tmp_path / 'dataset'
    inventory = tmp_path / 'inventory.csv'
    inventory.write_text('product_id,path\nbag_a,C:/data/gallery/bag_a/../../secret.jpg\n', encoding='utf8')
    with pytest.raises(ValueError, match='Invalid inventory'):
        read_inventory(inventory, root, 'gallery')


def test_relative_inventory_paths(tmp_path):
    root = tmp_path / 'data'
    file = root / 'gallery' / 'img 001' / 'front.jpg'
    file.parent.mkdir(parents=True)
    file.write_bytes(b'image')
    inventory = tmp_path / 'gallery_inventory.csv'
    inventory.write_text('product_id,path\nimg 001,gallery/img 001/front.jpg\n', encoding='utf-8')
    cached = read_inventory(inventory, root, 'gallery')
    assert_inventory_matches(scan_dataset_role(root, 'gallery'), cached, 'gallery')
