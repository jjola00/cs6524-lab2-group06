# Find the supplied IEEE class without moving report assets into the root.
use Config;
$ENV{'TEXINPUTS'} = './assets/latex//' . $Config{path_sep} . ($ENV{'TEXINPUTS'} // '');
