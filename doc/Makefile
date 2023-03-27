all: tex/lognormal.pdf

FIGS = figs/pk_pkG.pdf \
       figs/pkest_realspace_simNGP_estNGP_fxshift0.0_pkest.pdf \
       figs/pkest_realspace_simNGP_estNGP_fxshift0.5_pkest.pdf \
       figs/pkest_realspace_simCIC_estNGP_fxshift0.0_pkest.pdf \
       figs/pkest_realspace_simCIC_estNGP_fxshift0.5_pkest.pdf \
       \
       figs/pkest_redshiftspace_simCIC_estNGP_velo1_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo1_fxshift0.5_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo2_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo2_fxshift0.5_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo3_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo3_fxshift0.5_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo4_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo4_fxshift0.5_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo5_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo5_fxshift0.5_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo6_fxshift0.0_pkest.pdf \
       figs/pkest_redshiftspace_simCIC_estNGP_velo6_fxshift0.5_pkest.pdf \


tex/lognormal.pdf: tex/lognormal.tex references.bib $(FIGS)
	latexmk -cd -pdf -interaction=nonstopmode $<

figs/pk_pkG.pdf: scripts/plot_pk_pkG.jl
	./$<

figs/%_pkest.pdf: sim_configs/%.yml sim_results/%_pkest.tsv scripts/plot_pkest.jl
	./scripts/plot_pkest.jl $<

sim_results/%_pkest.tsv: sim_configs/%.yml scripts/calc_sims.jl
	./scripts/calc_sims.jl $<
