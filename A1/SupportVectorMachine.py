import numpy as np

class SoftSVM():

    def __init__(self, feature_size, C=1.0):

        self.C = C
        self.w = np.random.normal(0, 0.5, size=(feature_size,))
        self.b = np.random.normal(0, 0.1)

    def distance(self, X):
        return np.dot(X, self.w) + self.b

    def gradient(self, X, y, y_pred):

        mask = y * y_pred < 1
        y_mask = y[mask]

        dw = np.dot(y_mask, X[mask])
        dw = self.w - self.C * dw
        db = - self.C * np.sum(y_mask)
        return dw, db

    def loss(self, y, y_pred):
        hinge_loss = np.sum(np.maximum(0, 1 - y * y_pred))
        return 0.5 * np.sum(self.w**2) + self.C * hinge_loss

    def fit(self, X, y, learning_rate=0.001, epochs=1, batch_size=16):

        dlen = X.shape[0]
        indices = np.arange(dlen)
        train_loss = []

        for _ in range(epochs):

            np.random.shuffle(indices)
            for i in range(0, dlen, batch_size):

                indx = indices[i:i+batch_size]
                X_batch = X[indx]
                y_batch = y[indx]
                y_pred = self.distance(X_batch)

                dw, db = self.gradient(X_batch, y_batch, y_pred)
                self.w -= learning_rate * dw
                self.b -= learning_rate * db

            y_pred = self.distance(X)
            train_loss.append(self.loss(y, y_pred))

        return train_loss

    def predict(self, X):
        return np.sign(self.distance(X))
