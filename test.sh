# CUDA_VISIBLE_DEVICES=2 python test.py --opt_path ./options/test/DADM.yml


# dataset=(nh outdoor)
# # times=(30 40 60 70 80 90)
# times=(20 10 0)

# for set in ${dataset[*]}
# do
#     for t in ${times[*]}
#     do
#         # echo "DADM_"$set".yml"
#         path="./options/shell/DADM_"$set".yml"
#         # echo $path $t
#         CUDA_VISIBLE_DEVICES=1 python test.py --opt_path $path --times $t
#     done
# done


CUDA_VISIBLE_DEVICES=1 python test.py --opt_path ./options/test/DADM.yml --times 50

# CUDA_VISIBLE_DEVICES=3 python test.py --opt_path ./options/test/DADM.yml --times $50