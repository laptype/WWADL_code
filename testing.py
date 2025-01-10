
from init_utils import init_dataset, init_model, init_test_dataset
from pipeline.tester import Tester

def test(config):
    test_dataset = init_test_dataset(config)
    model = init_model(config)
    pt_file_name = config['testing'].get('pt_file_name', None)
    tester = Tester(config, test_dataset=test_dataset, model=model, pt_file_name=pt_file_name)
    tester.testing()