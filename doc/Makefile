all: tex/lognormal.pdf

tex/lognormal.pdf: tex/lognormal.tex references.bib figs/*
	latexmk -cd -pdf -interaction=nonstopmode $<

figs/pk_pkG.png: scripts/plots.jl
	./$<
