;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "margin=3cm") ("IEEEtrantools" "retainorgcmds") ("mathrsfs" "") ("inputenc" "utf8") ("amsthm" "") ("amsfonts" "") ("amssymb" "") ("amscd" "") ("multirow" "") ("booktabs" "") ("fullpage" "") ("lastpage" "") ("enumitem" "") ("fancyhdr" "") ("wrapfig" "") ("setspace" "") ("calc" "") ("multicol" "") ("cancel" "") ("amsmath" "") ("empheq" "") ("framed" "") ("tcolorbox" "most") ("xcolor" "table" "")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "amsmath"
    "amsthm"
    "amsfonts"
    "amssymb"
    "amscd"
    "multirow"
    "booktabs"
    "xcolor"
    "fullpage"
    "lastpage"
    "enumitem"
    "fancyhdr"
    "wrapfig"
    "setspace"
    "calc"
    "multicol"
    "cancel"
    "empheq"
    "framed"
    "tcolorbox")
   (LaTeX-add-labels
    "fig:placeholder")
   (LaTeX-add-lengths
    "tabcont")
   (LaTeX-add-amsthm-newtheorems
    "defn"
    "reg"
    "exer"
    "note")
   (LaTeX-add-xcolor-definecolors
    "shadecolor"))
 :latex)

