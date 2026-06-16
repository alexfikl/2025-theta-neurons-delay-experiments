PYTHON := "python -X dev"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc("Reformat all source code")]
format: isort black pyproject justfmt

[doc("Run ruff isort fixes over the source code")]
isort:
    ruff check --fix --select=I experiments/*.py
    ruff check --fix --select=RUF022 experiments/*.py
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc("Run ruff format over the source code")]
black:
    ruff format experiments/*.py
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run pyproject-fmt over the configuration")]
pyproject:
    {{ PYTHON }} -m pyproject_fmt \
        --indent 4 --max-supported-python "3.14" \
        pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc("Run just --fmt over the justfile")]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

[doc("Clean up the bibliography (with bibtex-tidy)")]
tidy:
    @bibtex-tidy \
        --modify \
        --sort \
        --sort-fields \
        --drop-all-caps \
        --merge last \
        --numeric \
        --strip-enclosing-braces \
        --trailing-commas \
        --blank-lines \
        --remove-empty-fields \
        --remove-dupe-fields \
        --wrap \
        ../bibliography.bib

# }}}
# {{{ linting

[doc("Run all linting checks over the source code")]
lint: typos ruff ty

[doc("Run typos over the source code and documentation")]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc("Run ruff checks over the source code")]
ruff:
    ruff check experiments/*.py
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run ty checks over the source code")]
ty:
    ty check experiments/*.py
    @echo -e "\e[1;32mty clean!\e[0m"

# }}}
# {{{ develop

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.11' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

[doc("Export PDF files to the figures folder")]
crop:
    #!/usr/bin/env bash
    set -euxo pipefail

    for pdf in *.pdf; do
        pdfcrop "${pdf}" "${pdf}"
    done

# }}}
# {{{ figures

[doc("Generate plot for Figure 1")]
figure1:
    {{ PYTHON }} experiments/type1-equilibria-and-tau-plots.py

[doc("Generate plot for Figure 2")]
figure2:
    {{ PYTHON }} experiments/type1-dirac-tau-and-l1-with-hopf-boundary.py

[doc("Generate plot for Figure 3, 4, 5")]
figure345:
    {{ PYTHON }} experiments/type2-center-regions-and-tau1-l1.py

[doc("Generate plot for Figure 6")]
figure6:
    {{ PYTHON }} experiments/attractor-classification.py

[doc("Generate plot for Figure 7")]
figure7:
    {{ PYTHON }} experiments/theta-neurons.py \
        --select set1 --npoints 4096 \
        --regions --trajectories \
        experiments/parameters-paper.toml

[doc("Generate plot for Figure 8")]
figure8:
    {{ PYTHON }} experiments/theta-neurons.py \
        --select set2 --npoints 4096 \
        --phase \
        experiments/parameters-paper.toml

[doc("Generate plot for Figure 9")]
figure9:
    {{ PYTHON }} experiments/theta-neurons.py \
        --select set31 --npoints 4096 \
        --regions --trajectories \
        experiments/parameters-paper.toml
    {{ PYTHON }} experiments/theta-neurons.py \
        --select set32 --npoints 4096 \
        --regions --trajectories \
        experiments/parameters-paper.toml

[doc("Generate plot for Figure 10")]
figure10:
    {{ PYTHON }} experiments/theta-neurons.py \
        --select set4 --npoints 4096 \
        --regions --trajectories \
        experiments/parameters-paper.toml

# }}}
