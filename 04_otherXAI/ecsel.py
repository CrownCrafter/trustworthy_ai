import numpy as np
from typing import Callable, Tuple, Optional, Dict, Any
import warnings
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
import torch
import random



def adam_minibatch(
    loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    theta0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    num_epochs: int = 10,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
    patience: int = 10,
    min_delta: float = 1e-6,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    
    """Adam optimizer for a parameter vector theta, using mini-batches."""

    if rng is None:
        rng = np.random.default_rng()
    
    theta = np.array(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    
    n = X.shape[0]
    beta1, beta2 = betas
    t = 0
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    best_theta = theta.copy()
    patience_counter = 0
    
    use_validation = X_val is not None and y_val is not None
    
    if verbose:
        if use_validation:
            print(f"Starting Adam: {n} train samples, {X_val.shape[0]} val samples, batch={batch_size}, lr={lr}")
        else:
            print(f"Starting Adam: {n} train samples, batch={batch_size}, lr={lr}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        
        epoch_train_loss = 0.0
        num_batches = 0
        
        # Training loop - only uses training data
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            Xb = X[batch_idx]
            yb = y[batch_idx]
            
            # Compute gradient on training batch
            g = grad_fn(theta, Xb, yb)
            
            # uncomment for gradient clipping            
            # Clip by norm to prevent exploding gradients
            max_grad_norm = 1.0
            grad_norm = np.linalg.norm(g)
            if grad_norm > max_grad_norm:
                g = g * (max_grad_norm / grad_norm)
                # if verbose and grad_norm > 10.0:  # Only warn for very large gradients
                #     print(f"    Gradient clipped: norm {grad_norm:.2f} -> {max_grad_norm}")
        
            if weight_decay != 0.0:
                g = g + weight_decay * theta
            
            if not np.all(np.isfinite(g)):
                if verbose:
                    print(f"Warning: Invalid gradients at step {t}")
                continue
            
            # Adam update
            t += 1
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Track training loss
            batch_loss = loss_fn(theta, Xb, yb)
            if np.isfinite(batch_loss):
                epoch_train_loss += batch_loss * len(batch_idx)
                num_batches += 1
        
        # Compute epoch losses
        if num_batches > 0:
            avg_train_loss = epoch_train_loss / n
        else:
            avg_train_loss = float('inf')
        
        train_losses.append(avg_train_loss)
        
        # Compute validation loss (monitoring only, not training)
        if use_validation:
            val_loss = loss_fn(theta, X_val, y_val)
            val_losses.append(val_loss)
            monitoring_loss = val_loss
            loss_type = "Val"
        else:
            monitoring_loss = avg_train_loss
            loss_type = "Train"
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if verbose:
            if num_epochs <= 5 or epoch == 0 or (epoch + 1) % (num_epochs // 5) == 0 or epoch == num_epochs - 1:
                if use_validation:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.6f}, Val={val_loss:.6f}, Time={epoch_time:.2f}s")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train_loss:.6f}, Time={epoch_time:.2f}s")
        
        # Early stopping based on validation (or training if no validation)
        if monitoring_loss < best_loss - min_delta:
            best_loss = monitoring_loss
            best_theta = theta.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} ({loss_type} loss stopped improving)")
            break
    
    # Return best parameters found
    info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": epoch + 1,
        "steps": t,
        "best_loss": best_loss,
        "converged": patience_counter >= patience,
        "used_validation": use_validation
    }
    
    return best_theta, info


class SignomialClassifier:
    def __init__(self, 
                 K=1, 
                 l1_strength=0.1, 
                 batch_size=256, 
                 lr=1e-3, 
                 num_epochs=50, 
                 n_restarts=1,
                 patience=10, 
                 min_delta=1e-6,
                 gradient_method='analytical', #or 'finite_diff'
                 use_sigmoid=None,  # None=auto (sigmoid for binary, softmax for multi), True=force sigmoid, False=force softmax
                 sigmoid_threshold=0.5,
                 random_state=None,
                 verbose=False,
                 internal_scaling_range=None):
        self.K = K
        self.l1_strength = l1_strength
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_restarts = n_restarts
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_method = gradient_method
        self.use_sigmoid = use_sigmoid
        self.sigmoid_threshold = sigmoid_threshold
        self.random_state = random_state
        self.verbose = verbose
        self.internal_scaling_range = internal_scaling_range


        
        if gradient_method not in ['analytical', 'finite_diff']:
            raise ValueError("gradient_method must be 'analytical' or 'finite_diff'")
        
        # Model state
        self.scaler_ = None
        self.label_encoder_ = None
        self.best_params_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.training_info_ = None
        self._is_fitted = False
        self._using_sigmoid = False  # Will be set during fit
    
    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, Z):
        """Numerically stable softmax."""
        Z = np.clip(Z, -500, 500)
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def _compute_predictions(self, params, X):
        """Compute signomial logits: z_c = sum_k [c_k * prod_j(x_j^alpha_kj)]"""
        n_samples, m = X.shape
        params_per_term = m + 1
        
        # For sigmoid (binary), we only compute one logit
        # For softmax (multi-class), we compute logits for all classes
        n_logits = 1 if self._using_sigmoid else self.n_classes_
        params_per_class = self.K * params_per_term
        
        Z = np.zeros((n_samples, n_logits))
        
        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                start_idx = class_start + k * params_per_term
                if start_idx + params_per_term > len(params):
                    continue
                
                const = params[start_idx]
                exponents = params[start_idx + 1:start_idx + params_per_term]
                
                X_safe = np.maximum(X, 1e-10)
                term_value = const * np.prod(np.power(X_safe, exponents), axis=1)
                
                if np.any(~np.isfinite(term_value)):
                    term_value = np.nan_to_num(term_value, nan=0.0, posinf=1e10, neginf=-1e10)
                
                Z[:, c] += term_value
        
        return Z
    
    def _loss_function(self, params, X_batch, y_batch):
        """Cross-entropy loss with L1 regularization."""
        try:
            Z = self._compute_predictions(params, X_batch)
            
            if self._using_sigmoid:
                # Binary classification with sigmoid
                P_pos = self._sigmoid(Z.ravel())
                P_pos = np.clip(P_pos, 1e-15, 1 - 1e-15)
                
                # Binary cross-entropy
                ce_loss = -np.mean(y_batch * np.log(P_pos) + (1 - y_batch) * np.log(1 - P_pos))
            else:
                # Multi-class classification with softmax
                P = self._softmax(Z)
                P = np.clip(P, 1e-15, 1 - 1e-15)
                
                n_samples = len(y_batch)
                y_onehot = np.zeros((n_samples, self.n_classes_))
                y_onehot[np.arange(n_samples), y_batch.astype(int)] = 1
                
                ce_loss = -np.mean(y_onehot * np.log(P)) # Dividing cross entopy by B!
            
            # L1 on both constants and exponents
            l1_penalty = self.l1_strength * np.sum(np.abs(params))
            
            return ce_loss + l1_penalty
            
        except:
            return 1e10
    
    def _analytical_gradient(self, params, X_batch, y_batch):
        """Analytical gradients with L1."""
        n_samples, m = X_batch.shape
        params_per_term = m + 1
        params_per_class = self.K * params_per_term
        
        Z = self._compute_predictions(params, X_batch)
        
        if self._using_sigmoid:
            # Binary classification with sigmoid
            P_pos = self._sigmoid(Z.ravel())
            
            # Gradient of binary cross-entropy w.r.t. logit
            dL_dz = (P_pos - y_batch) / n_samples  # Shape: (n_samples,)
            dL_dZ = dL_dz.reshape(-1, 1)  # Shape: (n_samples, 1)
            n_logits = 1
        else:
            # Multi-class classification with softmax
            P = self._softmax(Z)
            
            y_onehot = np.zeros((n_samples, self.n_classes_))
            y_onehot[np.arange(n_samples), y_batch.astype(int)] = 1
            
            dL_dZ = (P - y_onehot) / n_samples # Dividing by B!
            n_logits = self.n_classes_
        
        grad = np.zeros_like(params)
        X_safe = np.maximum(X_batch, 1e-10)
        
        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                start_idx = class_start + k * params_per_term
                if start_idx + params_per_term > len(params):
                    continue
                
                const = params[start_idx]
                exponents = params[start_idx + 1:start_idx + params_per_term]
                
                term_values = np.prod(np.power(X_safe, exponents), axis=1)
                
                # Gradient w.r.t. constant
                grad[start_idx] = np.sum(dL_dZ[:, c] * term_values)
                grad[start_idx] += self.l1_strength * np.sign(const)
                
                # Gradients w.r.t. exponents
                for j in range(m):
                    exp_grad = const * np.log(X_safe[:, j]) * term_values
                    grad[start_idx + 1 + j] = np.sum(dL_dZ[:, c] * exp_grad)
                    grad[start_idx + 1 + j] += self.l1_strength * np.sign(exponents[j])
        
        return grad
    
    def _finite_diff_gradient(self, params, X_batch, y_batch):
        """Finite difference gradients."""
        grad = np.zeros_like(params)
        h = 1e-8
        base_loss = self._loss_function(params, X_batch, y_batch)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            loss_plus = self._loss_function(params_plus, X_batch, y_batch)
            grad[i] = (loss_plus - base_loss) / h
        
        return grad
    
    def _gradient_function(self, params, X_batch, y_batch):
        """Route to correct gradient method."""
        if self.gradient_method == 'analytical':
            return self._analytical_gradient(params, X_batch, y_batch)
        else:
            return self._finite_diff_gradient(params, X_batch, y_batch)
    
    def _initialize_parameters(self, rng):
        """Initialize parameters with sparse random values."""
        m = self.n_features_
        params_per_term = m + 1
        params_per_class = self.K * params_per_term
        
        # For sigmoid (binary), only one set of parameters
        # For softmax (multi-class), parameters for all classes
        n_logits = 1 if self._using_sigmoid else self.n_classes_
        n_params = n_logits * params_per_class
        
        params = np.zeros(n_params)
        
        for c in range(n_logits):
            class_start = c * params_per_class
            for k in range(self.K):
                term_start = class_start + k * params_per_term
                
                # Small random constant
                # params[term_start] = rng.normal(0, 0.1) #original
                params[term_start] = rng.normal(1, 5)
                
                # Sparse random exponents
                for j in range(m):
                    if rng.random() < 0.3:
                        # params[term_start + 1 + j] = rng.normal(0, 0.5) #original
                        params[term_start + 1 + j] = rng.normal(0, 1)
        
        return params
    
    def fit(self, X, y, validation_split=0.0):
        """
        Fit the signomial classifier WITHOUT internal validation, 
        with reproducible multi-restart initialization.
        """
        start_time = time.time()

        # --- REPRODUCIBILITY FIX: Determine the base seed ---
        # Use the provided random_state, or a non-deterministic seed if None
        # We will use this base_seed + restart_index to seed each restart's RNG
        base_seed = self.random_state if self.random_state is not None else None
        
        # NOTE: The original code used a single evolving RNG (rng) here.
        # We remove it and create a new rng for each restart inside the loop.
        # rng = np.random.default_rng(self.random_state) if self.random_state is not None else np.random.default_rng()
        # print("uses batch size:", self.batch_size)
        X = np.array(X, dtype=float)
        y = np.array(y)

        # Ignore validation split completely
        X_train, y_train = X, y
        X_val, y_val = None, None

        if self.verbose:
            print(f"Using all {len(X_train)} samples for training (no internal validation)")

        # Fit preprocessing on the training data
        # self.scaler_ = MinMaxScaler(feature_range=(0.1, 1.01))
        # # self.scaler_ = MinMaxScaler(feature_range=(0.01, 10.1))
        # self.scaler_.fit(X_train)
        
        # --- Handle internal feature scaling ---
        if self.internal_scaling_range is not None:
            # Apply MinMax scaling inside the classifier
            self.scaler_ = MinMaxScaler(feature_range=self.internal_scaling_range)
            self.scaler_.fit(X_train)
            X_train_scaled = self.scaler_.transform(X_train)
            if self.verbose:
                print(f"[SignomialClassifier] Internal MinMax scaling applied with range={self.internal_scaling_range}")
        else:
            # No internal scaling — assume user handled scaling externally
            self.scaler_ = None
            X_train_scaled = X_train
            if self.verbose:
                print("[SignomialClassifier] No internal scaling (using raw/external scaled inputs)")


        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_train)

        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_train.shape[1]

        # Auto mode: sigmoid for binary, softmax for multi-class
        self._using_sigmoid = self.use_sigmoid if self.use_sigmoid is not None else (self.n_classes_ == 2)

        if self._using_sigmoid and self.n_classes_ != 2:
            raise ValueError("Sigmoid activation can only be used for binary classification")

        # Transform training data
        # X_train_scaled = self.scaler_.transform(X_train)
        y_train_encoded = self.label_encoder_.transform(y_train)

        best_params = None
        best_loss = float('inf')
        best_info = None

        for restart in range(self.n_restarts):
            if self.verbose and self.n_restarts > 1:
                print(f"\nRestart {restart + 1}/{self.n_restarts}")

            # --- REPRODUCIBILITY FIX: Create a new RNG for each restart ---
            # If a base_seed (random_state) is provided, seed the restart's RNG deterministically.
            if base_seed is not None:
                rng = np.random.default_rng(base_seed + restart) 
            else:
                # If no random_state is set, use a non-deterministic RNG
                rng = np.random.default_rng() 
            # ------------------------------------------------------------------

            initial_params = self._initialize_parameters(rng)

            try:
                # Train using Adam, WITHOUT internal validation
                params, info = adam_minibatch(
                    loss_fn=self._loss_function,
                    grad_fn=self._gradient_function,
                    theta0=initial_params,
                    X=X_train_scaled,
                    y=y_train_encoded,
                    X_val=None,        # <-- disable validation
                    y_val=None,        # <-- disable validation
                    batch_size=self.batch_size,
                    lr=self.lr,
                    num_epochs=self.num_epochs,
                    shuffle=True,
                    rng=rng, # Pass the restart-specific RNG
                    patience=self.patience,
                    min_delta=self.min_delta,
                    verbose=self.verbose
                )

                if info['best_loss'] < best_loss:
                    best_loss = info['best_loss']
                    best_params = params.copy()
                    best_info = info.copy()
                    if self.verbose and self.n_restarts > 1:
                        print(f"New best loss: {best_loss:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Restart {restart + 1} failed: {e}")
                continue

        if best_params is None:
            raise ValueError("All optimization restarts failed")

        self.best_params_ = best_params
        self.training_info_ = best_info
        self._is_fitted = True

        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Best loss: {best_loss:.6f}")
            print(f"Used validation: False")  # Always False now

        return self
        
    def fit_works(self, X, y, validation_split=0.0):
        """
        Fit the signomial classifier WITHOUT internal validation.
        """
        start_time = time.time()

        rng = np.random.default_rng(self.random_state) if self.random_state is not None else np.random.default_rng()

        X = np.array(X, dtype=float)
        y = np.array(y)

        # Ignore validation split completely
        X_train, y_train = X, y
        X_val, y_val = None, None

        if self.verbose:
            print(f"Using all {len(X_train)} samples for training (no internal validation)")

        # Fit preprocessing on the training data
        self.scaler_ = MinMaxScaler(feature_range=(0.1, 1.01))
        # self.scaler_ = MinMaxScaler(feature_range=(0.01, 10.1))
        self.scaler_.fit(X_train)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_train)

        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_train.shape[1]

        # Auto mode: sigmoid for binary, softmax for multi-class
        self._using_sigmoid = self.use_sigmoid if self.use_sigmoid is not None else (self.n_classes_ == 2)

        if self._using_sigmoid and self.n_classes_ != 2:
            raise ValueError("Sigmoid activation can only be used for binary classification")

        # Transform training data
        X_train_scaled = self.scaler_.transform(X_train)
        y_train_encoded = self.label_encoder_.transform(y_train)

        best_params = None
        best_loss = float('inf')
        best_info = None

        for restart in range(self.n_restarts):
            if self.verbose and self.n_restarts > 1:
                print(f"\nRestart {restart + 1}/{self.n_restarts}")

            initial_params = self._initialize_parameters(rng)

            try:
                # Train using Adam, WITHOUT internal validation
                params, info = adam_minibatch(
                    loss_fn=self._loss_function,
                    grad_fn=self._gradient_function,
                    theta0=initial_params,
                    X=X_train_scaled,
                    y=y_train_encoded,
                    X_val=None,        # <-- disable validation
                    y_val=None,        # <-- disable validation
                    batch_size=self.batch_size,
                    lr=self.lr,
                    num_epochs=self.num_epochs,
                    shuffle=True,
                    rng=rng,
                    patience=self.patience,
                    min_delta=self.min_delta,
                    verbose=self.verbose
                )

                if info['best_loss'] < best_loss:
                    best_loss = info['best_loss']
                    best_params = params.copy()
                    best_info = info.copy()
                    if self.verbose and self.n_restarts > 1:
                        print(f"New best loss: {best_loss:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"Restart {restart + 1} failed: {e}")
                continue

        if best_params is None:
            raise ValueError("All optimization restarts failed")

        self.best_params_ = best_params
        self.training_info_ = best_info
        self._is_fitted = True

        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Best loss: {best_loss:.6f}")
            print(f"Used validation: False")  # Always False now

        return self

    
    # def fit(self, X, y, validation_split=0.0):

    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X, dtype=float)
        
        # Use internal scaler if available
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X  # assume externally scaled or raw
        
        Z = self._compute_predictions(self.best_params_, X_scaled)
        
        if self._using_sigmoid:
            P_pos = self._sigmoid(Z.ravel())
            return np.column_stack([1 - P_pos, P_pos])
        else:
            return self._softmax(Z)

    
    def predict_proba_original(self, X):
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        Z = self._compute_predictions(self.best_params_, X_scaled)
        
        if self._using_sigmoid:
            # Binary classification: return probabilities for both classes
            P_pos = self._sigmoid(Z.ravel())
            return np.column_stack([1 - P_pos, P_pos])
        else:
            # Multi-class: return softmax probabilities
            return self._softmax(Z)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        # y_pred_encoded = np.argmax(proba, axis=1)
        
        if self._using_sigmoid:
            # threshold for binary classification
            y_pred_encoded = (proba[:, 1] >= self.sigmoid_threshold).astype(int)
        else:
            # Default: use argmax (threshold=0.5 for binary, standard for multi-class)
            y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def score(self, X, y):
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)
    
    def get_model_summary(self):
        """Get summary of fitted model."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        # Count active parameters
        total_params = len(self.best_params_)
        active_params = np.sum(np.abs(self.best_params_) > 0.01)
        sparsity = 1 - active_params / total_params
        
        return {
            'n_classes': self.n_classes_,
            'n_features': self.n_features_,
            'activation': 'sigmoid' if self._using_sigmoid else 'softmax',
            'n_terms_per_class': self.K,
            'total_parameters': total_params,
            'active_parameters': active_params,
            'sparsity': sparsity,
            'final_loss': self.training_info_['train_losses'][-1] if self.training_info_['train_losses'] else None,
            'converged': self.training_info_['converged'],
            'epochs_trained': self.training_info_['epochs']
        }
    
    def get_learned_formula(self, feature_names=None, threshold=0.01):
        """
        Get human-readable mathematical formula of the learned signomial model.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before extracting formula")
        
        if feature_names is None:
            feature_names = [f'X_{i}' for i in range(self.n_features_)]
        elif len(feature_names) != self.n_features_:
            raise ValueError(f"feature_names length ({len(feature_names)}) must match n_features ({self.n_features_})")
        
        m = self.n_features_
        params_per_term = m + 1
        params_per_class = self.K * params_per_term
        
        class_formulas = []
        
        if self._using_sigmoid:
            # Binary classification with sigmoid: only one formula
            terms = []
            
            for k in range(self.K):
                start_idx = k * params_per_term
                if start_idx + params_per_term > len(self.best_params_):
                    continue
                
                const = self.best_params_[start_idx]
                exponents = self.best_params_[start_idx + 1:start_idx + params_per_term]
                
                # Skip terms with very small constants
                if abs(const) < threshold:
                    continue
                
                # Build term components
                term_parts = []
                for j, exp in enumerate(exponents):
                    if abs(exp) > threshold:
                        if abs(exp - 1.0) < threshold:
                            term_parts.append(feature_names[j])
                        else:
                            term_parts.append(f"{feature_names[j]}^{exp:.2f}")
                
                # Construct full term
                if term_parts:
                    if abs(const - 1.0) < threshold:
                        term_str = " * ".join(term_parts)
                    else:
                        term_str = f"{const:.2f} * " + " * ".join(term_parts)
                    terms.append(term_str)
                elif abs(const) > threshold:  # Constant term only
                    terms.append(f"{const:.2f}")
            
            # Build formula
            if terms:
                if len(terms) == 1:
                    formula = f"z = {terms[0]}"
                else:
                    # Handle signs for multiple terms
                    formula_parts = [terms[0]]
                    for term in terms[1:]:
                        if term.startswith('-'):
                            formula_parts.append(f" - {term[1:]}")
                        else:
                            formula_parts.append(f" + {term}")
                    formula = f"z = " + "".join(formula_parts)
            else:
                formula = f"z = 0"
            
            class_formulas.append(formula)
            
            # Final sigmoid formula
            final_formula = f"\nP({self.classes_[1]}) = sigmoid(z) = 1 / (1 + exp(-z))\n"
            
        else:
            # Multi-class classification with softmax: formula for each class
            for c, class_name in enumerate(self.classes_):
                class_start = c * params_per_class
                terms = []
                
                for k in range(self.K):
                    start_idx = class_start + k * params_per_term
                    if start_idx + params_per_term > len(self.best_params_):
                        continue
                    
                    const = self.best_params_[start_idx]
                    exponents = self.best_params_[start_idx + 1:start_idx + params_per_term]
                    
                    # Skip terms with very small constants
                    if abs(const) < threshold:
                        continue
                    
                    # Build term components
                    term_parts = []
                    for j, exp in enumerate(exponents):
                        if abs(exp) > threshold:
                            if abs(exp - 1.0) < threshold:
                                term_parts.append(feature_names[j])
                            else:
                                term_parts.append(f"{feature_names[j]}^{exp:.2f}")
                    
                    # Construct full term
                    if term_parts:
                        if abs(const - 1.0) < threshold:
                            term_str = " * ".join(term_parts)
                        else:
                            term_str = f"{const:.2f} * " + " * ".join(term_parts)
                        terms.append(term_str)
                    elif abs(const) > threshold:  # Constant term only
                        terms.append(f"{const:.2f}")
                
                # Build class formula
                if terms:
                    if len(terms) == 1:
                        class_formula = f"z_{class_name} = {terms[0]}"
                    else:
                        # Handle signs for multiple terms
                        formula_parts = [terms[0]]
                        for term in terms[1:]:
                            if term.startswith('-'):
                                formula_parts.append(f" - {term[1:]}")
                            else:
                                formula_parts.append(f" + {term}")
                        class_formula = f"z_{class_name} = " + "".join(formula_parts)
                else:
                    class_formula = f"z_{class_name} = 0"
                
                class_formulas.append(class_formula)
            
            # Final softmax formula
            logit_names = [f"z_{class_name}" for class_name in self.classes_]
            final_formula = f"\nP(class) = softmax([{', '.join(logit_names)}])\n"
        
        return "\n".join(class_formulas) + final_formula
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'K': self.K,
            'l1_strength': self.l1_strength,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'num_epochs': self.num_epochs,
            'n_restarts': self.n_restarts,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'gradient_method': self.gradient_method,
            'use_sigmoid': self.use_sigmoid,
            'sigmoid_threshold': self.sigmoid_threshold,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self