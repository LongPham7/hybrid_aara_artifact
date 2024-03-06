# The config.mk file must contain cruft
# to build against Clp
# (set LIBDIRS and INCDIRS)
include config/clp_paths.mk

include config/Makefile

.PHONY: all clean partialclean force configs runtime


CAMLC=ocamlfind ocamlc
CAMLOPT=ocamlfind ocamlopt

OCB_FLAGS=-use-ocamlfind -no-hygiene -I utils -I parsing -I typing -I clp -I automatic_differentiation -I volesti -I probabilistic_programming -I raml -I hybrid_aara
OCB=INCDIRS='$(INCDIRS)' LIBDIRS='$(LIBDIRS)' ocamlbuild $(OCB_FLAGS)


RAML_RUNTIME=raml_runtime/rprob.ml raml_runtime/rnat.ml raml_runtime/rarray.ml raml_runtime/raml.ml raml_runtime/pervasives.ml
RAML_RUNTIME_CMI=$(RAML_RUNTIME:.ml=.cmi)

MAIN=main

TEST=unit_test

all: $(MAIN) $(MAIN).opt $(TEST) $(TEST).opt runtime

clean: partialclean
	$(OCB) -clean

force: ;

configs: raml/rpath.ml utils/config.ml


# main

$(MAIN): force configs
	$(OCB) $(MAIN).byte
	mv $(MAIN).byte $(MAIN)

partialclean::
	rm -f $(MAIN)

$(MAIN).opt: force configs
	$(OCB) $(MAIN).native
	mv $(MAIN).native $(MAIN).opt

partialclean::
	rm -f $(MAIN).opt

# this will run gen-runime three times in a parallel make
$(RAML_RUNTIME_CMI): $(RAML_RUNTIME) $(MAIN)
	./$(MAIN) gen-runtime

runtime: $(RAML_RUNTIME_CMI)


# Unit tests

$(TEST): force configs
	$(OCB) $(TEST).byte
	mv $(TEST).byte $(TEST)

partialclean::
	rm -f $(TEST)

$(TEST).opt: force configs
	$(OCB) $(TEST).native
	mv $(TEST).native $(TEST).opt

partialclean::
	rm -f $(TEST).opt


# The configuration file

raml/rpath.ml: raml/rpath.mlp config/Makefile
	@rm -f raml/rpath.ml
	sed -e 's|%%DESTDIR%%|$(shell pwd)|' \
	    raml/rpath.mlp > raml/rpath.ml
	@chmod -w raml/rpath.ml

utils/config.ml: utils/config.mlp config/Makefile
	@rm -f utils/config.ml
	sed -e 's|%%LIBDIR%%|$(shell $(CAMLC) -config | awk '{if ($$1 ~ "standard_library:") print $$2}')|' \
	    -e 's|%%BYTERUN%%|$(shell $(CAMLC) -config | awk '{if ($$1 ~ "standard_runtime:") print $$2}')|' \
	    -e 's|%%HOST_CMI_MAGIC_NUMBER%%|$(shell $(CAMLC) -config | awk '{if ($$1 ~ "cmi_magic_number:") print $$2}')|' \
	    -e 's|%%CCOMPTYPE%%|cc|' \
	    -e 's|%%BYTECC%%|$(BYTECC) $(BYTECCCOMPOPTS) $(SHAREDCCCOMPOPTS)|' \
	    -e 's|%%NATIVECC%%|$(NATIVECC) $(NATIVECCCOMPOPTS)|' \
	    -e 's|%%PACKLD%%|$(PACKLD)|' \
	    -e 's|%%BYTECCLIBS%%|$(BYTECCLIBS)|' \
	    -e 's|%%NATIVECCLIBS%%|$(NATIVECCLIBS)|' \
	    -e 's|%%RANLIBCMD%%|$(RANLIBCMD)|' \
	    -e 's|%%ARCMD%%|$(ARCMD)|' \
	    -e 's|%%CC_PROFILE%%|$(CC_PROFILE)|' \
	    -e 's|%%ARCH%%|$(ARCH)|' \
	    -e 's|%%MODEL%%|$(MODEL)|' \
	    -e 's|%%SYSTEM%%|$(SYSTEM)|' \
	    -e 's|%%EXT_OBJ%%|.o|' \
	    -e 's|%%EXT_ASM%%|.s|' \
	    -e 's|%%EXT_LIB%%|.a|' \
	    -e 's|%%EXT_DLL%%|.so|' \
	    -e 's|%%SYSTHREAD_SUPPORT%%|$(SYSTHREAD_SUPPORT)|' \
	    -e 's|%%ASM%%|$(ASM)|' \
	    -e 's|%%ASM_CFI_SUPPORTED%%|$(ASM_CFI_SUPPORTED)|' \
	    -e 's|%%WITH_FRAME_POINTERS%%|$(WITH_FRAME_POINTERS)|' \
	    -e 's|%%MKDLL%%|$(MKDLL)|' \
	    -e 's|%%MKEXE%%|$(MKEXE)|' \
	    -e 's|%%MKMAINDLL%%|$(MKMAINDLL)|' \
	    utils/config.mlp > utils/config.ml
	@chmod -w utils/config.ml

partialclean::
	rm -f utils/config.ml raml/rpath.ml
