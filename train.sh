mkdir -p checkpoints/
mkdir -p Snapshots/

chmod +x checkpoints
chmod +x Snapshots

python3 train.py --epochs 100 --store_path ./Snapshots/