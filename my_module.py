import jax
import jax.numpy as jnp
import haiku as hk

class MirrorInvariantNetwork(hk.Module):
    def __init__(self, n_hiddens, n_actions, return_all=False):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions
        # Define shared linear layers
        self.hidden_layer = hk.Linear(self.n_hiddens, b_init=hk.initializers.TruncatedNormal(1.))
        self.output_layer = hk.Linear(self.n_actions, b_init=hk.initializers.TruncatedNormal(1.))

        self.return_all = return_all

    def __call__(self, net_input, input_mot=None):
        # Apply shared hidden layer with tanh activation to both original and mirrored inputs
        mirrored_input = net_input[:, ::-1]

        # Process the original input
        if not self.return_all:
            next_state_original = jax.nn.tanh(self.hidden_layer(net_input if input_mot is None else jnp.concatenate([net_input, input_mot], axis=-1)))
        else:
            next_state_original = jax.nn.tanh(self.hidden_layer(net_input if input_mot is None else jnp.concatenate([net_input, jnp.tile(input_mot, 9)[..., jnp.newaxis]], axis=-1)))
        next_gist_original = self.output_layer(next_state_original)

        # Process the mirrored input using the same layers
        if not self.return_all:
            next_state_mirrored = jax.nn.tanh(self.hidden_layer(mirrored_input if input_mot is None else jnp.concatenate([mirrored_input, input_mot], axis=-1)))
        else:
            next_state_mirrored = jax.nn.tanh(self.hidden_layer(mirrored_input if input_mot is None else jnp.concatenate([mirrored_input, jnp.tile(input_mot, 9)[..., jnp.newaxis]], axis=-1)))
        next_gist_mirrored = self.output_layer(next_state_mirrored)

        # Average the results to ensure symmetry
        next_gist = (next_gist_original + next_gist_mirrored[:, ::-1]) / 2
        return next_gist

def mirror_invariant_network(net_input, n_hiddens=8, addon=None):
    model = MirrorInvariantNetwork(n_hiddens=n_hiddens, n_actions=3)
    return model(net_input, addon)

if __name__ == "__main__":
    # Initialize the network
    rng_key = jax.random.PRNGKey(42)
    contrast = jax.random.normal(rng_key, (1, 3))
    network = hk.without_apply_rng(hk.transform(mirror_invariant_network))

    # Use the network
    def use_network(params, net_input, addon=None):
        return network.apply(params, net_input, addon=addon)
    
    # binary reward test
    reward = jnp.array(1.).reshape(1, -1)

    params = network.init(rng_key, jax.random.normal(rng_key, (1, 2)))

    action = jnp.array(0).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    action = jnp.array(1).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))


    reward = jnp.array(-1.).reshape(1, -1)
    action = jnp.array(0).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    action = jnp.array(1).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))

    print("Actual actions")
    params = network.init(rng_key, jax.random.normal(rng_key, (1, 4)))

    action = jnp.array([0., 0., 1.]).reshape(1, -1)
    reward = jnp.array(-1.).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    reward = jnp.array(1).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))

    action = jnp.array([0., 1., 0.]).reshape(1, -1)
    reward = jnp.array(-1.).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    reward = jnp.array(1).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))

    action = jnp.array([1., 0., 0.]).reshape(1, -1)
    reward = jnp.array(-1.).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    reward = jnp.array(1).reshape(1, -1)
    print("Result for rewarded:", use_network(params, action, addon=reward))
    quit()


    params = network.init(rng_key, contrast)

    # Test the network
    test_contrast = jnp.array([1.0, 2.0, 3.0]).reshape(1, -1)
    result = use_network(params, test_contrast)
    print("Result for [1, 2, 3]:", result)

    mirrored_contrast = jnp.array([3.0, 2.0, 1.0]).reshape(1, -1)
    mirrored_result = use_network(params, mirrored_contrast)
    print("Result for [3, 2, 1]:", mirrored_result)

    # Test the network with additional input

    contrast = jax.random.normal(rng_key, (1, 5))
    network = hk.without_apply_rng(hk.transform(mirror_invariant_network))
    params = network.init(rng_key, contrast)

    result = use_network(params, test_contrast, addon=jnp.array([0.1, 0.2]).reshape(1, -1))
    print("Result for [1, 2, 3]:", result)

    mirrored_result = use_network(params, mirrored_contrast, addon=jnp.array([0.1, 0.2]).reshape(1, -1))
    print("Result for [3, 2, 1]:", mirrored_result)