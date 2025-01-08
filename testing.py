
from init_utils import init_dataset, init_model, init_test_dataset
from pipeline.tester import Tester

def test(config):
    test_dataset = init_test_dataset(config)
    model = init_model(config)
    tester = Tester(config, test_dataset=test_dataset, model=model)
    tester.testing()