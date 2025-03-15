import os 
import sys
import zipfile
from tqdm import tqdm

category_name = 'Shoe'
except_list = ['Sperry_TopSider_pSUFPWQXPp3', 'Sperry_TopSider_tNB9t6YBUf3', 
        'UGG_Bailey_Button_Triplet_Womens_Boots_Black_7','UGG_Bailey_Bow_Womens_Clogs_Black_7',
        'Tory_Burch_Kiernan_Riding_Boot', 'UGG_Bailey_Button_Womens_Boots_Black_7', 
        'UGG_Cambridge_Womens_Black_7', 'UGG_Classic_Tall_Womens_Boots_Chestnut_7',
        'UGG_Classic_Tall_Womens_Boots_Grey_7','UGG_Jena_Womens_Java_7', 'W_Lou_z0dkC78niiZ',
        'Womens_Hikerfish_Boot_in_Black_Leopard_bVSNY1Le1sm','Womens_Hikerfish_Boot_in_Black_Leopard_ridcCWsv8rW',
        'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_imlP8VkwqIH',
        'Womens_Hikerfish_Boot_in_Linen_Leather_Sparkle_Suede_QktIyAkonrU',
        'Rayna_BootieWP','ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange']

data_path = 'data'

file_list = os.listdir(data_path)
model_list = []
for file_name in file_list:
    name, ext = os.path.splitext(file_name)
    if ext == '.zip' and name not in except_list:
        model_list.append(name)

cate_model_list = []

for model_name in tqdm(model_list):
    zip_file_path = os.path.join(data_path, f'{model_name}.zip')
    zip_file = zipfile.ZipFile(zip_file_path, 'r')
    data_str = str(zip_file.read('metadata.pbtxt'))
    idx = data_str.find('categories') + len('categories')
    data_str = data_str[idx:]
    idx1 = data_str.find('{') + 1
    idx2 = data_str.find('}')
    data_str = data_str[idx1:idx2]
    idx1 = data_str.find('\"') + 1
    idx2 = data_str.find('\"', idx1)
    data_str = data_str[idx1:idx2]
    data_str = data_str.strip()
    if data_str == category_name:
        cate_model_list.append(model_name)

save_path = f'{category_name}_file_list.txt'
with open(save_path, 'w') as f:
    for model_name in cate_model_list:
        f.write(f'{model_name}\n')

out_path = f'{category_name}_data'

# cate_model_list = ['11pro_SL_TRX_FG']

for model_name in cate_model_list:
    zip_file_path = os.path.join(data_path, f'{model_name}.zip')
    zip_file = zipfile.ZipFile(zip_file_path, 'r')
    for file_name in zip_file.namelist():
        # if file_name == ''
        extract_path = os.path.join(out_path, model_name)
        zip_file.extract(file_name, path=extract_path)
        print(model_name, file_name)
    mat_path = os.path.join(extract_path, 'materials/textures')
    mat_png_n = len(os.listdir(mat_path))
    assert(mat_png_n == 1)
    mat_png_path = os.path.join(mat_path, 'texture.png')
    mat_png_path_new = os.path.join(extract_path, 'meshes', 'texture.png')
    cmd = f'mv {mat_png_path} {mat_png_path_new}'
    print(cmd)
    os.system(cmd)
    cmd = 'rm -r %s' % os.path.join(extract_path, 'materials')
    print(cmd)
    os.system(cmd)
    # break