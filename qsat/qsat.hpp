#include <vector>

namespace qsat {
  
  typedef int Variable;

  class Literal {
  public:
    Literal(Variable var, bool sign = false);
    
  private:
    int literalId;
  };

  class Clause {
  public:
    Clause(std::vector<Literal>& lits);

  private:
    std::vector<Literal> literals;
  };



}





