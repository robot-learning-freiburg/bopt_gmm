import numpy as np

from bopt_gmm.gmm import GMM


if __name__=='__main__':
    gmm = GMM.load_model('models/gmm/sd_3p/n00_f.npy')

    print(gmm.semantic_dims())
    print(gmm.semantic_inference_weights(('position', 'force')))
    print(gmm.semantic_prediction_weights(('velocity',), ('position', 'force')))

    lol = gmm.calculate_reweighting_inference_update(np.array([[0.5, 0.5],
                                                               [0.2, 0.8],
                                                               [0.8, 0.2]]), ('position', 'force'))
    gmm2 = gmm.update_gaussian(sigma_scale=lol)
    print(gmm2.semantic_inference_weights(('position', 'force')))

    foo = gmm.calculate_reweighting_prediction_update({'velocity': np.array([[0.5, 0.5],
                                                                             [0.2, 0.8],
                                                                             [0.8, 0.2]])}, ('position', 'force'))

    class GMM3DPredictZ(GMM):
        def __init__(self, priors, means, cvar):
            super().__init__(priors=priors, means=means, cvar=cvar)

        def predict(self, xy):
            return super().predict(xy, (0, 1))

        def predictX(self, yz):
            return super().predict(yz, (1, 2))

    g1 = GMM(priors=np.array([1, 1, 1]), 
             means=np.array([[1, 2, 3]]).T, 
             cvar=np.array([[[0.1]]]*3))
    w1 = g1.get_weights(np.array([[1, 2, 3]]).T)
    print(f'Weights for 1, 2, 3:\n{w1}')

    g2 = GMM3DPredictZ(priors=np.array([1,  1,  1]),
                       means=np.array([[1,  1,  1],
                                       [4,  5, -2],
                                       [9, -2,  4]]),
                       cvar=np.stack([np.eye(3)]*3, axis=0))
    
    w2 = g2.get_weights(np.array([[1,  1,  1],
                                  [4,  5, -2],
                                  [9, -2,  4]]))
    print(f'Weights for 3d case:\n{w2}')

    w3 = g2.get_weights(np.array([[1,  1],
                                  [4,  5],
                                  [9, -2]]), dims=(0, 1))
    print(f'Weights for 3d case without Z:\n{w3}')

    w4 = g2.get_weights(np.array([[6.5, 1.5, 1]]))
    print(f'3d point perfectly between 2 and 3:\n{w4}')

    p1 = g2.predict(np.array([[4, 5]]))
    print(f'Prediction for Z at (4, 5): {p1}')
    print(f'Prediction for Z at (9, 2): {g2.predict(np.array([[9, -2]]))}')
