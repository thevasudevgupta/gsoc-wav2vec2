import tfx.v1 as tfx
import argparse


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_root: str,
    module_file: str,
    metadata_path: str,
    num_train_steps: int,
    num_eval_steps: int,
):

    input_config = tfx.proto.Input(splits=[
        tfx.proto.Input.Split(name="train", pattern="train-clean-100/*.tfrecord"),
        tfx.proto.Input.Split(name="eval", pattern="dev-clean/*.tfrecord")
    ])

    example_gen = tfx.components.ImportExampleGen(input_base=data_root, input_config=input_config)
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs["examples"],
        train_args=tfx.proto.TrainArgs(num_steps=num_train_steps),
        eval_args=tfx.proto.TrainArgs(num_steps=num_eval_steps),
    )

    components = [example_gen, trainer]
    metadata_connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata_connection_config,
        components=components,
    )


def get_parser():
    parser = argparse.ArgumentParser()
    # TODO: add args
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    pipeline = create_pipeline(
        pipeline_name=args.pipeline_name,
        pipeline_root=args.pipeline_root,
        data_root=args.data_root,
        module_file=args.module_file,
        metadata_path=args.metadata_path,
        num_train_steps=args.num_train_steps,
        num_eval_steps=args.num_eval_steps,
    )

    tfx.orchestration.LocalDagRunner().run(pipeline)
