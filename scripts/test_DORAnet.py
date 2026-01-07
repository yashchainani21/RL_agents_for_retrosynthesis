import doranet.modules.enzymatic as enzymatic
import doranet.modules.synthetic as synthetic
import doranet.modules.post_processing as post_processing

user_starters = {'OC1C=CC=CC1'}

user_helpers = {'O','O=O','[H][H]','O=C=O','C=O','[C-]#[O+]','Br','[Br][Br]','CO',
                'C=C','O=S(O)O','N','O=S(=O)(O)O','O=NO','N#N','O=[N+]([O-])O','NO',
                'C#N','S','O=S=O','N#CO'}

user_target = {'OC1=CC=CC=C1'}        

job_name = "test_benzene_partial_hydrogenation"

# forward_network = enzymatic.generate_network(
#     job_name = job_name,
#     starters = user_starters,
#     gen = 1,
#     direction = "forward",
#     )

forward_network = synthetic.generate_network(
    job_name = job_name,
    starters = user_starters,
    helpers = user_helpers,
    gen = 1,
    direction = "forward",
    )

for mol in forward_network.mols:
    print(mol.uid)


post_processing.one_step(
    networks = {
        forward_network,
        },
    total_generations = 1,
    starters = user_starters,
    helpers = user_helpers,
    target = user_target,
    job_name = job_name,
    )
