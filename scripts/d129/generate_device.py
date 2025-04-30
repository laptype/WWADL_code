import itertools

# 原始的配置列表
receivers_to_keep_list = [
    {},
    {'imu': ['rh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '1'},
    {'imu': ['lh'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '2'},
    {'imu': ['rp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '3'},
    {'imu': ['lp'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '4'},
    {'imu': None, 'wifi': False, 'airpods': True, 'channel': 6, 'modality': 'imu', 'tag': '5'},
    {'imu': ['gl'], 'wifi': False, 'airpods': False, 'channel': 6, 'modality': 'imu', 'tag': '6'},
    {'imu': None, 'wifi': True, 'airpods': False, 'channel': 270, 'modality': 'wifi', 'tag': '7'},

    
    {'imu': ['rh'], 'wifi': True, 'airpods': False, 'channel': (6, 270), 'modality': 'wifimu', 'tag': '8'},
    {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '9'},
    {'imu': ['rp'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '10'},
    {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '11'},
    {'imu': ['rh', 'gl'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '12'},
    {'imu': ['rh', 'rp'], 'wifi': True, 'airpods': False, 'channel': (12, 270), 'modality': 'wifimu', 'tag': '13'},
    {'imu': ['rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '14'},

    {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '15'},
    {'imu': ['rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '16'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': True, 'airpods': False, 'channel': (18, 270), 'modality': 'wifimu', 'tag': '17'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '18'},
    {'imu': ['rh', 'rp', 'gl'], 'wifi': True, 'airpods': True, 'channel': (24, 270), 'modality': 'wifimu', 'tag': '19'},
    {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': False, 'airpods': True, 'channel': 36, 'modality': 'imu', 'tag': '20'},
    {'imu': ['rh', 'rp', 'lh', 'lp', 'gl'], 'wifi': True, 'airpods': True, 'channel': (36, 270), 'modality': 'imu', 'tag': '21'}, 

    {'imu': ['lh', 'rh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '23'},
    {'imu': ['lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '24'},
    {'imu': ['lh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '25'},
    {'imu': ['lh', 'lp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '26'},
    {'imu': ['gl', 'lh'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '27'},
    {'imu': ['lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '28'},
    {'imu': ['gl', 'lp'], 'wifi': False, 'airpods': False, 'channel': 12, 'modality': 'imu', 'tag': '29'},
    {'imu': ['lh', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '30'},
    {'imu': ['lh', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '31'},
    {'imu': ['gl', 'lh', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '32'},
    {'imu': ['lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '33'},
    {'imu': ['gl', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '34'},
    {'imu': ['lh', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '35'},
    {'imu': ['gl', 'lh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '36'},
    {'imu': ['gl', 'lh', 'lp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '37'},
    {'imu': ['gl', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 18, 'modality': 'imu', 'tag': '38'},
    {'imu': ['lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '39'},
    {'imu': ['gl', 'lh', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '40'},
    {'imu': ['gl', 'lh', 'lp', 'rh'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '41'},
    {'imu': ['gl', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '42'},
    {'imu': ['gl', 'lh', 'lp', 'rp'], 'wifi': False, 'airpods': False, 'channel': 24, 'modality': 'imu', 'tag': '43'},
    {'imu': ['gl', 'lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': False, 'channel': 30, 'modality': 'imu', 'tag': '44'},

    {'imu': ['rh'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '45'},
    {'imu': ['lh'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '46'},
    {'imu': ['lp'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '47'},
    {'imu': ['gl'], 'wifi': False, 'airpods': True, 'channel': 12, 'modality': 'imu', 'tag': '48'},
    {'imu': ['lh', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '49'},
    {'imu': ['lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '50'},
    {'imu': ['gl', 'rh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '51'},
    {'imu': ['lh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '52'},
    {'imu': ['lh', 'lp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '53'},
    {'imu': ['gl', 'lh'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '54'},
    {'imu': ['lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '55'},
    {'imu': ['gl', 'lp'], 'wifi': False, 'airpods': True, 'channel': 18, 'modality': 'imu', 'tag': '56'},
    {'imu': ['lh', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '57'},
    {'imu': ['lh', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '58'},
    {'imu': ['gl', 'lh', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '59'},
    {'imu': ['lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '60'},
    {'imu': ['gl', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '61'},
    {'imu': ['lh', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '62'},
    {'imu': ['gl', 'lh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '63'},
    {'imu': ['gl', 'lh', 'lp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '64'},
    {'imu': ['gl', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 24, 'modality': 'imu', 'tag': '65'},
    {'imu': ['lh', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '66'},
    {'imu': ['gl', 'lh', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '67'},
    {'imu': ['gl', 'lh', 'lp', 'rh'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '68'},
    {'imu': ['gl', 'lp', 'rh', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '69'},
    {'imu': ['gl', 'lh', 'lp', 'rp'], 'wifi': False, 'airpods': True, 'channel': 30, 'modality': 'imu', 'tag': '70'},
]

# # 获取所有IMU设备的选项
# imu_devices = ['rh', 'lh', 'rp', 'lp', 'gl']

# # 提取已有的 imu、wifi 和 airpods 配置，并对 imu 列表排序
# existing_configs = [
#     {'imu': sorted(config['imu']), 'wifi': config['wifi'], 'airpods': config['airpods']}
#     for config in receivers_to_keep_list if config.get('imu') is not None
# ]

# # 用来存放新生成的配置
# new_receivers = []

# # 生成不同的排列组合
# for r in range(1, len(imu_devices) + 1):
#     # 使用 itertools 生成所有可能的组合（不区分顺序）
#     for combo in itertools.combinations(imu_devices, r):
#         new_config = {
#             'imu': sorted(list(combo)),  # 确保 imu 是排序的
#             'wifi': False,  # 固定为 False
#             'airpods': True,  # 固定为 False
#             'channel': len(combo) * 6 + 6,  # 根据 imu 数量计算通道数
#             'modality': 'imu',
#             'tag': str(len(receivers_to_keep_list) + len(new_receivers) + 1)
#         }
#         # 检查是否有重复的 imu、wifi 和 airpods 配置
#         if not any(
#             existing['imu'] == new_config['imu'] and
#             existing['wifi'] == new_config['wifi'] and
#             existing['airpods'] == new_config['airpods']
#             for existing in existing_configs
#         ):
#             new_receivers.append(new_config)

# # 输出新生成的配置
# for config in new_receivers:
#     print(config)
