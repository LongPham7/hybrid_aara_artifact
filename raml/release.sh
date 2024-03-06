#!/bin/sh

VERSION=1.5.0
DIR=raml-$VERSION

function execute() {
  MSG="$1"
  CMD="$2"

  /bin/echo -n "$MSG..."
  if $CMD; then
    /bin/echo " done"
  else
    /bin/echo " failed"
    exit 1
  fi
}

function copy_from_git() {
  rm -rf $DIR
  git archive --format=tar --prefix=$DIR/ HEAD | tar xf -
}

execute "Copying files into $DIR with git archive" copy_from_git

execute "Removing QPL licensed files" "rm -rf $DIR/parsing $DIR/typing $DIR/utils"

cd $DIR
execute "Patching config file" "patch" < config.patch
cd ".."

function further_clean_up() {
  mv $DIR/Makefile $DIR/Makefile.in
  rm -f $DIR/release.sh $DIR/.gitignore $DIR/config.patch
}

execute "Further clean up" further_clean_up

execute "Downloading ocaml-4.01.0 source file from caml.inria.fr" \
        "curl --silent -o $DIR/ocaml-4.01.0.tar.gz http://caml.inria.fr/pub/distrib/ocaml-4.01/ocaml-4.01.0.tar.gz"

execute "Creating ${DIR}.tar.gz..." "tar zcf $DIR.tar.gz $DIR"
