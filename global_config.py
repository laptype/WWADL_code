import json

basic_config_path = '/home/lanbo/WWADL/WWADL_code/basic_config.json'

with open(basic_config_path, 'r') as json_file:
    config = json.load(json_file)

def write_setting(data, save_path=''):
    # path = os.path.join(save_path, 'setting.json')
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
