import json


class SettingManager:
    config = {}

    @staticmethod
    def default_config():
        return {
            "replace_track_box": False,
            "filter_box_on_the_edge": False,
        }

    @staticmethod
    def load_config():
        try:
            fr = open('./settings.json')
            SettingManager.config = json.load(fr)
        except FileNotFoundError:
            fw = open('./settings.json', 'w')
            SettingManager.config = SettingManager.default_config()
            json.dump(SettingManager.config, fw)
            fw.close()

    @staticmethod
    def modify_config(config):
        SettingManager.config = config
        fw = open('./settings.json', 'w')
        json.dump(SettingManager.config, fw)
        fw.close()
