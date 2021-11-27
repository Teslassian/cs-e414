#==============SCATTER PLOT ====================================
# TODO: INSERT CODE FOR ABOVE INSTRUCTION HERE
def get_signer_single_signature_signal(signer_num, num_features, single_sign_index):
    sign = read_signatures('../resources/data/signatures/sign'+str(signer_num)+'/*.txt')
    single_sign = sign[single_sign_index]
    single_sign_signal = sign_norm(single_sign[:num_features, :])
    return single_sign_signal

#Only 1, 4 is working when using 3 training signals
#Only 1, 4, 5 is working when using 4 training signals
#Only 1, 3, 4, 5 is working when using 5 training signals and 7 states
signer_num = 3 # [1..5] 
single_sign_index = 9 # [0..14]index 0 1 2 was used for training
single_sign_signal = get_signer_single_signature_signal(signer_num,2,single_sign_index)
signatory_hmm_model = sign_models[signer_num -1]
print(single_sign_signal.shape)
print(signatory_hmm_model.trans.shape)

seq, ll = signatory_hmm_model.viterbi(single_sign_signal)
print(ll)

plt.scatter(single_sign_signal[0,:],single_sign_signal[1,:], s= 6, c=seq)

#==============CONFUSUION MATRIX=================================
#ground truth: 60 correct signature labels for each signal [1..5]
#newSigs: 12 signal arrays for every signer
#         every signal consists of 2 arrays. One for X and one for Y
#testSigs: 12 x 5 signals
for sig in range(len(testsigs)):
    ll_max = -10000000000000000000000000000000000000
    signer_max = -1
    for signer in range(5):
        seq, ll = sign_models[signer].viterbi(testsigs[sig])
        ll = ll/len(testsigs[sig])
        if ll > ll_max:
            ll_max = ll
            signer_max = signer
    predict += [signer_max+1]