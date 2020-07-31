from options import Options
from data import load_data
from rlkg_model import RLKGModel


if __name__ == "__main__":

    option = Options().parse()

    train_loader, test_loader, poi_info, user_KG = load_data(option)

    model = RLKGModel(poi_info,user_KG,option)

    model.fit(train_loader)

    model.evaluate(test_loader)





