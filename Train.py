from ModelLoader import ModelLoader
from networks.mesonet import Meso4, MesoInception4

fake = "dataset/real_vs_fake/real-vs-fake/train/fake"
real = "dataset/real_vs_fake/real-vs-fake/train/fake"
model = ModelLoader(model=Meso4, model_path="models/model.pth")


def main():
    model.train(fake_path=fake, real_path=real, batch_size=1, num_epochs=100, iteration_nums=50000)
    # model.save_model("models/model.pth")


if __name__ == "__main__":
    main()
