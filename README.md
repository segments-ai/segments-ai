<p align="center">
    <br>
        <img src="assets/logo_no_shadow-with_text-blue_background.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/segments-ai/segments-ai/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/segments-ai/segments-ai.svg?color=blue">
    </a>
    <!-- <a href="https://github.com/segments-ai/segments-ai/actions">
        <img alt="Tests" src="https://github.com/segments-ai/segments-ai/actions/workflows/tests.yml/badge.svg">
    </a> -->
    <a href="https://segments-python-sdk.readthedocs.io/en/latest/?badge=latest">
        <img alt="Documentation" src="https://readthedocs.org/projects/segments-python-sdk/badge/?version=latest">
    </a>
    <!-- <a href="https://github.com/segments-ai/segments-ai/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/segments-ai/segments-ai.svg">
    </a> -->
    <a href="https://github.com/segments-ai/segments-ai/releases">
        <img alt="Downloads" src="https://img.shields.io/pypi/dm/segments-ai">
    </a>
</p>

[Segments.ai](https://segments.ai/) is the training data platform for computer vision engineers and labeling teams. Our powerful labeling interfaces, easy-to-use management features, and extensive API integrations help you iterate quickly between data labeling, model training and failure case discovery.

![](assets/overview.png)

## Quickstart

Walk through [the Python SDK quickstart](https://docs.segments.ai/tutorials/python-sdk-quickstart).

## Documentation

Please refer to [the documentation](http://segments-python-sdk.rtfd.io/) for usage instructions.

## Blog

Read [our blog posts](https://segments.ai/blog) to learn more about the platform.

## Changelog

The most notable changes in v1.0 of the Python SDK compared to v0.73 include:

- Added Python type hints and better auto-generated docs.
- Improved error handling: functions now raise proper exceptions.
- New functions for managing issues and collaborators.

You can upgrade to v1.0 with `pip install -—upgrade segments-ai`. Please be mindful of following breaking changes:

- The client functions now return classes instead of dicts, so you should access properties using dot-based indexing (e.g. `dataset.description`) instead of dict-based indexing (e.g. `dataset[’description’]`).
- Functions now consistently raise exceptions, instead of sometimes silently failing with a print statement. You might want to handle these exceptions with a try-except block.
- Some legacy fields are no longer returned: `dataset.tasks`, `dataset.task_readme`, `dataset.data_type`.
- The default value of the `id_increment` argument in `utils.export_dataset()` and `utils.get_semantic_bitmap()` is changed from 1 to 0.
- Python 3.6 and lower are no longer supported.
