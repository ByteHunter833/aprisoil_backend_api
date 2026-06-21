from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf_v1 = tf.compat.v1

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = MODULE_DIR / "sandy_loam_nod.csv"


class PhysicsInformedNN:
    def __init__(
        self,
        t: np.ndarray,
        z: np.ndarray,
        theta: np.ndarray,
        layers_psi: list[int],
        layers_theta: list[int],
        layers_K: list[int],
    ) -> None:
        self.t = t
        self.z = z
        self.theta = theta
        self.layers_psi = layers_psi
        self.layers_theta = layers_theta
        self.layers_K = layers_K

        self.weights_psi, self.biases_psi = self.initialize_NN(layers_psi)
        self.weights_theta, self.biases_theta = self.initialize_MNN(layers_theta)
        self.weights_K, self.biases_K = self.initialize_MNN(layers_K)

        self.sess = tf_v1.Session(
            config=tf_v1.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
            )
        )

        self.z_tf = tf_v1.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf_v1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf_v1.placeholder(tf.float32, shape=[None, self.theta.shape[1]])

        (
            self.theta_pred,
            self.psi_pred,
            self.K_pred,
            self.f_pred,
            _,
            _,
            _,
            _,
        ) = self.net(self.t_tf, self.z_tf)

        self.loss = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred)) + tf.reduce_mean(
            tf.square(self.f_pred)
        )
        self.optimizer_Adam = tf_v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        self.sess.run(tf_v1.global_variables_initializer())

    def xavier_init(self, size: list[int]) -> tf.Variable:
        in_dim, out_dim = size[0], size[1]
        stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=stddev),
            dtype=tf.float32,
        )

    def initialize_NN(self, layers: list[int]) -> tuple[list[tf.Tensor], list[tf.Variable]]:
        weights: list[tf.Tensor] = []
        biases: list[tf.Variable] = []
        for layer_index in range(len(layers) - 1):
            weights.append(self.xavier_init([layers[layer_index], layers[layer_index + 1]]))
            biases.append(tf.Variable(tf.zeros([1, layers[layer_index + 1]], dtype=tf.float32)))
        return weights, biases

    def initialize_MNN(self, layers: list[int]) -> tuple[list[tf.Tensor], list[tf.Variable]]:
        weights: list[tf.Tensor] = []
        biases: list[tf.Variable] = []
        for layer_index in range(len(layers) - 1):
            weights.append(self.xavier_init([layers[layer_index], layers[layer_index + 1]]) ** 2)
            biases.append(tf.Variable(tf.zeros([1, layers[layer_index + 1]], dtype=tf.float32)))
        return weights, biases

    def net_psi(
        self,
        x: tf.Tensor,
        weights: list[tf.Tensor],
        biases: list[tf.Variable],
    ) -> tf.Tensor:
        hidden = x
        for layer_index in range(len(weights) - 1):
            hidden = tf.tanh(tf.add(tf.matmul(hidden, weights[layer_index]), biases[layer_index]))
        return -tf.exp(tf.add(tf.matmul(hidden, weights[-1]), biases[-1]))

    def net_theta(
        self,
        x: tf.Tensor,
        weights: list[tf.Tensor],
        biases: list[tf.Variable],
    ) -> tf.Tensor:
        hidden = x
        for layer_index in range(len(weights) - 1):
            hidden = tf.tanh(tf.add(tf.matmul(hidden, weights[layer_index]), biases[layer_index]))
        return tf.sigmoid(tf.add(tf.matmul(hidden, weights[-1]), biases[-1]))

    def net_K(
        self,
        x: tf.Tensor,
        weights: list[tf.Tensor],
        biases: list[tf.Variable],
    ) -> tf.Tensor:
        hidden = x
        for layer_index in range(len(weights) - 1):
            hidden = tf.tanh(tf.add(tf.matmul(hidden, weights[layer_index]), biases[layer_index]))
        return tf.exp(tf.add(tf.matmul(hidden, weights[-1]), biases[-1]))

    def net(
        self,
        t: tf.Tensor,
        z: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x = tf.concat([t, z], 1)
        psi = self.net_psi(x, self.weights_psi, self.biases_psi)
        log_h = tf.math.log(-psi)
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        k_value = self.net_K(-log_h, self.weights_K, self.biases_K)

        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        k_z = tf.gradients(k_value, z)[0]
        residual = theta_t - k_z * psi_z - k_value * psi_zz - k_z
        return theta, psi, k_value, residual, theta_t, psi_z, psi_zz, k_z

    def train(self, iterations: int, verbose: bool = False) -> list[float]:
        tf_dict = {self.t_tf: self.t, self.z_tf: self.z, self.theta_tf: self.theta}
        losses: list[float] = []

        for iteration in range(iterations):
            self.sess.run(self.train_op_Adam, tf_dict)
            if verbose and (iteration % 200 == 0 or iteration == iterations - 1):
                loss_value = float(self.sess.run(self.loss, tf_dict))
                losses.append(loss_value)
                print(f"Iteration: {iteration}, loss: {loss_value:.3e}")

        if not losses:
            losses.append(float(self.sess.run(self.loss, tf_dict)))
        return losses

    def predict(self, t_star: np.ndarray, z_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tf_dict = {self.t_tf: t_star, self.z_tf: z_star}
        theta_pred, psi_pred = self.sess.run([self.theta_pred, self.psi_pred], tf_dict)
        return theta_pred, psi_pred

    def close(self) -> None:
        self.sess.close()


def _resolve_data_path(hydrus: str, data_path: Optional[str | os.PathLike[str]] = None) -> Path:
    if data_path is not None:
        return Path(data_path)

    local_path = MODULE_DIR / f"{hydrus}_nod.csv"
    if local_path.exists():
        return local_path

    legacy_path = Path.cwd() / "Node_Inf" / "hydrus_nod_files" / f"{hydrus}_nod.csv"
    return legacy_path


def _validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.str.strip()
    required_columns = {"time", "depth", "theta"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training data is missing columns: {missing}")
    return data


def main_loop(
    hydrus: str,
    depth_increment: int,
    noise: float,
    num_layers_psi: int,
    num_neurons_psi: int,
    num_layers_theta: int,
    num_neurons_theta: int,
    num_layers_K: int,
    num_neurons_K: int,
    number_random: int,
    *,
    data_path: Optional[str | os.PathLike[str]] = None,
    train_iterations: int = 1000,
    verbose: bool = False,
) -> tuple[PhysicsInformedNN, pd.DataFrame]:
    tf_v1.reset_default_graph()
    tf_v1.set_random_seed(number_random)
    random.seed(number_random)
    np.random.seed(number_random)

    resolved_path = _resolve_data_path(hydrus, data_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Training data file was not found: {resolved_path}")

    data = _validate_training_data(pd.read_csv(resolved_path))

    t = data["time"].values[:, None]
    z = data["depth"].values[:, None]
    theta = data["theta"].values[:, None]

    z_star = np.hstack((t, z))
    theta_star = theta.flatten()[:, None]

    layers_psi = np.concatenate([[2], num_neurons_psi * np.ones(num_layers_psi), [1]]).astype(int).tolist()
    layers_theta = np.concatenate([[1], num_neurons_theta * np.ones(num_layers_theta), [1]]).astype(int).tolist()
    layers_K = np.concatenate([[1], num_neurons_K * np.ones(num_layers_K), [1]]).astype(int).tolist()

    fixed_position_full = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]
    fixed_position = fixed_position_full[:: max(depth_increment, 1)]

    depth_column = "zeta" if "zeta" in data.columns else "depth"
    fixed_list = data.index[
        np.round(data[depth_column], 3).isin(np.round(fixed_position, 3))
    ].values

    if len(fixed_list) == 0:
        raise ValueError(f"Training sample is empty for {hydrus}. Check depth/zeta values.")

    theta_train = theta_star[fixed_list, :] + noise * np.random.randn(len(fixed_list), 1)
    t_train, z_train = z_star[fixed_list, 0:1], z_star[fixed_list, 1:2]

    model = PhysicsInformedNN(t_train, z_train, theta_train, layers_psi, layers_theta, layers_K)
    model.train(train_iterations, verbose=verbose)
    return model, data


if __name__ == "__main__":
    trained_model, training_data = main_loop(
        hydrus="sandy_loam",
        depth_increment=1,
        noise=0,
        num_layers_psi=8,
        num_neurons_psi=40,
        num_layers_theta=1,
        num_neurons_theta=10,
        num_layers_K=1,
        num_neurons_K=10,
        number_random=111,
        data_path=DEFAULT_DATA_PATH,
        verbose=True,
    )
    theta_pred, _ = trained_model.predict(
        training_data["time"].values[:, None],
        training_data["depth"].values[:, None],
    )
    mse = float(np.mean((training_data["theta"].values[:, None] - theta_pred) ** 2))
    print(f"Training MSE: {mse:.6e}")
    trained_model.close()
