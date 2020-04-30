
#--------------------------------------------------------------------------
# Print more or less formatted output to console or file.
#
#--------------------------------------------------------------------------

spacer <- '\n###########################\n'

print_header <- function(txt_header) {
  txt_out <- str_c(spacer, txt_header, spacer, '\n')
  cat(txt_out)
}