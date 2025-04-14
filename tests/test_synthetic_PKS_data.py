import pickle

def test_number_of_LM_PKS_designs():
    with open('../data/raw/PKS_designs_and_products_LM.pkl', 'rb') as f:
        PKS_designs_and_bound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_bound_products_dict) == 29

    with open('../data/interim/PKS_designs_and_unbound_products_LM.pkl', 'rb') as f:
        PKS_designs_and_unbound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_unbound_products_dict) == 29

def test_number_of_unique_LM_products_without_stereochemistry():
    with open('../data/interim/unique_PKS_products_no_stereo_LM.txt','r') as f:
        products_list = f.readlines()
        assert len(products_list) == 31

def test_number_of_M1_PKS_designs():
    with open('../data/raw/PKS_designs_and_products_M1.pkl', 'rb') as f:
        PKS_designs_and_bound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_bound_products_dict) == 2784

    with open('../data/interim/PKS_designs_and_unbound_products_M1.pkl', 'rb') as f:
        PKS_designs_and_unbound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_unbound_products_dict) == 2784

def test_number_of_unique_M1_products_without_stereochemistry():
    with open('../data/interim/unique_PKS_products_no_stereo_M1.txt','r') as f:
        products_list = f.readlines()
        assert len(products_list) == 1535

def test_number_of_M2_PKS_designs():
    with open('../data/raw/PKS_designs_and_products_M2.pkl', 'rb') as f:
        PKS_designs_and_bound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_bound_products_dict) == 264509

    with open('../data/interim/PKS_designs_and_unbound_products_M2.pkl', 'rb') as f:
        PKS_designs_and_unbound_products_dict = pickle.load(f)

    assert len(PKS_designs_and_unbound_products_dict) == 264509

def test_number_of_unique_M2_products_without_stereochemistry():
    with open('../data/interim/unique_PKS_products_no_stereo_M2.txt','r') as f:
        products_list = f.readlines()
        assert len(products_list) == 67192

def test_number_of_M3_PKS_designs():
    with open('../data/raw/PKS_designs_and_products_M3.pkl', 'rb') as f:
        PKS_designs_and_bound_products_dict = pickle.load(f)
        assert len(PKS_designs_and_bound_products_dict) == 25128384

def test_number_of_unique_M3_products_without_stereochemistry():
    with open('../data/interim/unique_PKS_products_no_stereo_M3.txt','r') as f:
        products_list = f.readlines()
        assert len(products_list) == 2850334