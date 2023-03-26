all: lognormal.pdf

lognormal.pdf: lognormal.tex references.bib figs/*
	latexmk -pdf -interaction=nonstopmode $<

figs/pk_pkG.png: scripts/plots.jl
	./$<
