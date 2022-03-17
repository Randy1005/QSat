#include <vector>
#include <string>

namespace qsat {

  typedef int Variable;

  class Literal {
    public:
      Literal(Variable var, bool sign = false);

      int literalId;
  };

  class Clause {
    public:
      Clause(std::vector<Literal>& lits);

      std::vector<Literal> literals;
  };

  class Solver {
    public: 
      Solver();
      void ParseDimacs(const std::string& inputFileName);
      void Dump(std::ostream& os) const;

    private:
      void ReadClause(std::istringstream& in, std::vector<Literal>& lits);
      bool AddClause(std::vector<Literal>& lits);
      std::vector<Clause> clauses; 
  };



}





