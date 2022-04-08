
std::unordered_map<std::string, bool> solution {
  "dimacs_1", true, 

}

int main(int argc, char *argv[]) {

  auto gold = solution[input_file];

  auto mysol = qsat.solve(input_file);

  if(gold != mysol) std::exit(EXIT_FAILURE);

  return 0;

}
