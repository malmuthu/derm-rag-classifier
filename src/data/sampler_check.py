from collections import Counter
from dataset import get_dataloaders, CLASSES

def main():
    train_loader, _, _ = get_dataloaders(batch_size=32, use_weighted_sampler=True)

    label_counts = Counter()
    for _, labels in train_loader:
        label_counts.update(labels.tolist())

    total = sum(label_counts.values())
    print("Class distribution with weighted sampler active (one epoch):")
    for i, cls in enumerate(CLASSES):
        count = label_counts.get(i, 0)
        print(f"  {cls:6s} {count:5d}  ({count/total*100:5.1f}%)")

if __name__ == "__main__":
    main()