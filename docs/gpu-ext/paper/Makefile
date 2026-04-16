# Simple Makefile to build LaTeX paper with minted
# Usage: make (build), make clean, make distclean

TEX=main.tex
PDF=main.pdf
TEX_SRCS=$(wildcard tex/*.tex) $(TEX)
IMG_SRCS=$(wildcard img/*)
BIB=cite.bib

# SVG processing
SVGS=$(wildcard img/*.svg)
SVG_PDFS=$(SVGS:.svg=.pdf)

LATEX=pdflatex
LATEXFLAGS=-shell-escape -interaction=nonstopmode -halt-on-error
BIBTEX=bibtex

.PHONY: all clean distclean

all: $(PDF)

# Rule to convert SVG to PDF
img/%.pdf: img/%.svg
	convert $< $@

$(PDF): $(TEX_SRCS) $(IMG_SRCS) $(BIB) $(SVG_PDFS)
	$(LATEX) $(LATEXFLAGS) $(TEX)
	$(BIBTEX) main || true
	$(LATEX) $(LATEXFLAGS) $(TEX)
	$(LATEX) $(LATEXFLAGS) $(TEX)

clean:
	@rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.lof *.lot *.bcf *.run.xml
	@rm -rf _minted-*
	@rm -f $(SVG_PDFS)

distclean: clean
	@rm -f $(PDF)
