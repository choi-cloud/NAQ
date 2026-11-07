"""Episode pools. Call order (and hence RNG consumption) follows zip exp015/exp016 main():
seed -> load_data -> add_diffusion -> (selection embedding) -> ProtoNet -> train_pool -> valid/test pool.

Note: task_generator.downstream_task_generator loads ./save/downstream_tasks/<ds>/<n>way_<k>shot.pkl
when it exists (no RNG consumed); otherwise it samples and writes that file.
"""

from task_generator import downstream_task_generator, proto_task_generator


def build_train_pool(args, data):
    return [proto_task_generator(args, data) for _ in range(args.epoch)]


def build_eval_pools(args, data):
    return downstream_task_generator(args, data, args.n_way, args.k_shot_test, args.q_query_test)
