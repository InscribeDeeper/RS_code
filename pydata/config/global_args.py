
def get_folder_setting():
    files_folder = "../local_data/"
    output_folder = "../processed_data/"
    PATH_CLICK = files_folder + 'JD_click_data.csv'
    PATH_USER = files_folder + 'JD_user_data.csv'
    PATH_SKU = files_folder + 'JD_sku_data.csv'
    PATH_ORDER = files_folder + 'JD_order_data.csv'
    return files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER


if __name__ == '__main__':
    sys_init_date = "2018-03-13"
    files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER = get_folder_setting()
    # output = [files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER]
    print("\n".join([files_folder, output_folder, PATH_CLICK, PATH_USER, PATH_SKU, PATH_ORDER]))
