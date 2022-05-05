#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <chrono>
#include <unordered_map>
#include <filesystem>
#include <tao/pegtl.hpp>
namespace pegtl = tao::pegtl;

namespace qsat {

enum class Status {
  FALSE = 0,
  TRUE  = 1,
  UNDEFINED
};

/*
enum class VarType {
  ENUM = 0,
  ENUM_SS = 1,
  DEC_NUM,
  UNSPECIFIED
};
*/

struct ClauseSatisfiability {
  int clause_id;
  bool is_modified;
  int lit_id;
};

/**
@struct pegtl variable prefix
a single lower case v
*/
struct var_prefix : pegtl::one<'v'> {};

/**
@struct pegtl digits
e.g. 0, 12, 99, 12384, etc.
*/
struct digits : pegtl::plus<pegtl::digit> {};

/**
@struct pegtl variable name
consists of prefix and digits
*/
struct var_name : pegtl::seq<var_prefix, digits> {};

/**
@struct pegtl enum prefix
a single lower case s
*/
struct enum_prefix : pegtl::one<'s'> {};


/**
@struct pegtl enum name
format: s[digits]_[digits]
*/
struct enum_name : pegtl::seq<enum_prefix, digits, pegtl::one<'_'>, digits> {};

/**
@struct pegtl enum_names
concatenated with commas
*/
struct enum_names : pegtl::seq<pegtl::star<pegtl::space>, enum_name, 
                              pegtl::star<pegtl::space>,
                              pegtl::one<','>,
                              pegtl::star<pegtl::space>, 
                              enum_name,
                              pegtl::star<pegtl::space>,
                              pegtl::star<pegtl::one<','>,
                                          pegtl::star<pegtl::space>,
                                          enum_name>> {};

/**
@struct pegtl enum ss (don't know what this is) name
format: ss_[digits]
*/
struct enum_ss_name : pegtl::seq<pegtl::string<'s', 's'>, 
                                pegtl::one<'_'>, 
                                digits> {};

/**
@struct pegtl enum ss names
@brief enum ss names concatenated with commas
*/
struct enum_ss_names : pegtl::seq<pegtl::star<pegtl::space>, enum_ss_name, 
                                pegtl::star<pegtl::space>,
                                pegtl::one<','>,
                                pegtl::star<pegtl::space>, 
                                enum_ss_name,
                                pegtl::star<pegtl::space>, 
                                pegtl::star<pegtl::one<','>,
                                            pegtl::star<pegtl::space>,
                                            enum_ss_name>> {};

/**
@struct pegtl decimal number
@brief format: [digits]['d][digits]
*/
struct dec_num : pegtl::seq<digits, pegtl::string<'\'', 'd'>, digits> {};

/**
@struct pegtl var type
*/
struct var_type : pegtl::sor<enum_name, 
                            enum_ss_name, 
                            dec_num, 
                            pegtl::one<'*'>> {};

/**
@struct pegtl dash var type
@brief just so we don't do anything when we match the var_type
or else a duplicated enum would be written into z3
*/
struct dash_var_type : pegtl::seq<pegtl::star<pegtl::space>,
                                pegtl::one<'-'>,
                                pegtl::star<pegtl::space>,
                                var_type> {};

/**
@struct pegtl digits preceded with spaces (guaranteed to be plain numbers) 
@brief format:  [space*][digits], [digits] represent bits for BitVec
*/
struct digits_bits : pegtl::seq<pegtl::star<pegtl::space>,
                                          digits> {};

/**
@struct pegtl main variable table grammar
*/
struct var_table_grammar : pegtl::seq<var_name, dash_var_type, 
                                    pegtl::sor<enum_names, 
                                              enum_ss_names,
                                              digits_bits>> {};
/**
@struct variable state struct to categorize parsed elements during parse time
*/
struct var_state {
  std::string var_name;
  std::string enum_sort_name; // it's just the uppercase var_name
  std::vector<std::string> enum_names;
  std::vector<std::string> enum_ss_names;

  size_t bits; 
  // TODO: the bits member is based on guessing from the large task file...
  // will need to confirm with Yi Ling
  // ... may have more to store
};



/**
@struct action template
*/
template<typename Rule>
struct action {};


/**
@struct dash_var_type action
@brief when we match a enum here, it needs to be ignored 
(pop it off the state enum member) 
*/
template<>
struct action<dash_var_type> {
  template<typename ActionInput>
  static void apply(const ActionInput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {    
    // std::cout << "matched dash var type " << in.string() << std::endl;
    // pop the unnecessary enum name
    if (!state.enum_names.empty()) {
      state.enum_names.pop_back();
    }

    if (!state.enum_ss_names.empty()) {
      state.enum_ss_names.pop_back();
    }
  }
};




/**
@struct pegtl var_name action
@brief when we match the var_name grammar, store the var_name, 
store the enum sort name, and clear enum names
*/
template<>
struct action<var_name> {
  template<typename ActionInput>
  static void apply(const ActionInput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {    
    // std::cout << "var_name: " << in.string() << std::endl;
    state.var_name = in.string();
    
    std::string s(state.var_name);
    s[0] = toupper(s[0]);
    state.enum_sort_name = s;
    
    state.enum_names.clear();
    state.enum_ss_names.clear();
  }
};

/**
@struct pegtl enum_name action
@brief when we match the enum_name grammar, store them into
the enum name string vector
*/
template<>
struct action<enum_name> {
  template<typename actioninput>
  static void apply(const actioninput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {
    // std::cout << "i saw a enum_name." << std::endl;
    state.enum_names.push_back(in.string());
  }
};

/**
@struct pegtl enum_ss_name action
@brief when we match the enum_name grammar, store them into
the enum name string vector
*/
template<>
struct action<enum_ss_name> {
  template<typename actioninput>
  static void apply(const actioninput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {
    // std::cout << "i saw a enum_ss_name." << std::endl;
    state.enum_ss_names.push_back(in.string());
  }
};


// we probably don't need this
// since we can parse the actual enums or digits
// and know what we should write into z3
/*
template<>
struct action<var_type> {
  template<typename ActionInput>
  static void apply(const ActionInput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {
    // std::cout << "I saw a var_type." << std::endl;
  }
};
*/


/**
@struct enum_names action
@brief when we match the enum_names grammar,
write EnumSort(...) into z3
*/
template<>
struct action<enum_names> {
  template<typename ActionInput>
  static void apply(const ActionInput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {
    
    // EnumSort definition
    ofs << state.enum_sort_name << ", (";
    const size_t enum_names_size = state.enum_names.size();
    for (size_t i = 0; i < enum_names_size; i++) {
      ofs << state.enum_names[i];
      if (i != enum_names_size - 1) {
        ofs << ", ";
      }
    }
    ofs << ") = EnumSort(\'" << state.enum_sort_name << "\', [";
    for (size_t i = 0; i < enum_names_size; i++) {
      ofs << "\'" << state.enum_names[i] << "\'";
      if (i != enum_names_size - 1) {
        ofs << ", ";
      }
    }
    ofs << "])\n";

    // declare the actual enum sort in z3
    ofs << state.var_name << " = Const(\"" << state.var_name << "\", "
        << state.enum_sort_name << ")\n";
  }
};

template<>
struct action<enum_ss_names> {
  template<typename ActionInput>
  static void apply(const ActionInput& in,
                    var_state& state,
                    std::ofstream& ofs) {
    // EnumSort definition
    ofs << state.enum_sort_name << ", (";
    const size_t enum_ss_names_size = state.enum_ss_names.size();
    for (size_t i = 0; i < enum_ss_names_size; i++) {
      ofs << state.enum_ss_names[i];
      if (i != enum_ss_names_size - 1) {
        ofs << ", ";
      }
    }
    ofs << ") = EnumSort(\'" << state.enum_sort_name << "\', [";
    for (size_t i = 0; i < enum_ss_names_size; i++) {
      ofs << "\'" << state.enum_ss_names[i] << "\'";
      if (i != enum_ss_names_size - 1) {
        ofs << ", ";
      }
    }
    ofs << "])\n";

    // declare the actual enum sort in z3
    ofs << state.var_name << " = Const(\"" << state.var_name << "\", "
        << state.enum_sort_name << ")\n";
  }
};


/**
@struct pegtl digits_bits action
@brief store these as bits whenever we match the digit_bits grammar
*/
template<>
struct action<digits_bits> {
  template<typename ActionInput>
  static void apply(const ActionInput& in, 
                    var_state& state, 
                    std::ofstream& ofs) {

    // write BitVec definition to z3
    ofs << state.var_name << " = BitVec(\'" << state.var_name
        << "\', " << stoi(in.string()) << ")\n"; 
  }
};


/**
@struct Literal
@brief struct to create a literal

A literal is created from a given integer variable based on the following
encoding method:

v is positive => id = 2|v| - 2 => assignment id/2
v is negative => id = 2|v| - 1 => assignment id/2

var => id => assignment
 1  => 0  => 0/2 = 0
-1  => 1  => 1/2 = 0
 2  => 2  => 2/2 = 1
-2  => 3  => 3/2 = 1
 3  => 4  => 4/2 = 2
-3  => 5  => 5/2 = 2
...
*/
class Literal {
  
  friend struct Clause;
  friend class Solver;

  public:
    /**
    @brief constructs a literal with a given variable
    */
    Literal(int var);

  private:
    size_t _id;

};

/**
@struct Clause
@brief struct to create a clause
*/
struct Clause {
  
  Clause() = default;
  Clause(const Clause&) = default;
  Clause(Clause&&) = default;

  /**
  @brief constructs a clause with given literals using copy semantics
  */
  Clause(const std::vector<Literal>& lits);

  Clause(std::vector<Literal>&& lits);

  /**
  @brief default copy assignment operator
  */
  Clause& operator=(const Clause& rhs) = delete;

  /**
  @brief default move assignment operator
  */
  Clause& operator=(Clause&& rhs) = default;

  std::vector<Literal> literals;
};

/**
@class Solver
@brief a class to create a SAT solver object
*/
class Solver {
public: 
  /**
  @brief constructs a solver object
  */
  Solver();

  /**
  @brief reads in dimacs cnf file, and store the literals and clauses
  @param inputFileName the dimacs cnf file name
  */
  void read_dimacs(const std::string& inputFileName);
  
  /**
  @brief dumps the solver info via std::ostream
  @code{.cpp}
  solver.dump(std::cout); // dumps the solver info to standard output

  std::ofstream ofs("info.dmp"); 
  solver.dump(ofs); // dumps the solver info to the file "info.dmp"
  @endcode

  @param os output stream to dump info to
  */
  void dump(std::ostream& os) const;

  /**
  @brief solves the given cnf expression

  Determine whether the given cnf is satisfiable or not.
  If satisfiable, also construct a solution for the user
  @returns true if the input cnf is satisfiable, otherwise return false
  */
  bool solve();

  /**
  @brief a getter method for the stored clauses
  */
  const std::vector<Clause>& clauses() const; 
  
  /**
  @brief adds a clause given a vector of literals (using move semantics)
  */
  void add_clause(std::vector<Literal>&& lits);

  /**
  @brief adds a clause given a vector of literals (using copy semantics)
  */
  void add_clause(const std::vector<Literal>& lits);

  size_t num_clauses() const   { return _clauses.size(); }
  size_t num_variables() const { return _assignments.size(); }

  // v is positive => id = 2|v| - 2 => assignment id/2
  // v is negative => id = 2|v| - 1 => assignment id/2
  Status assignment_of(int variable) const {
    return _assignments[variable > 0 ? variable - 1 : -variable - 1];
  }

  void reset();
  void read_dimacs(std::istream&);


  bool transpile_task_to_z3(const std::string& task_file_name);
  bool transpile_task_to_dimacs(const std::string& task_file_name);

private:

  /**
  @brief utility method that reads in a parsed symbol, encode into a literal and store it
  @param[in] in the parsed symbol
  @param[out] lits the literal vector to store all the encoded literals (for each clause) 
  */
  void _read_clause(int symbol, std::vector<Literal>& lits);
  
  /*
  bool _dpll(std::vector<Clause>& clauses);
  void _unit_propagate(std::vector<Clause>& clauses);
  bool _has_unit_clause(std::vector<Clause>& clauses, size_t& unitClauseIndex);
  void _determine_literal(std::vector<Clause>& clauses, int new_lit_id);
  */

  bool _backtrack(int decision_depth, std::vector<Status>& assignments);
  bool _evaluate_clauses(const std::vector<Status>& assignments) ;
  size_t _propagate_constraint(int decision_depth, const std::vector<Status>& assignments);
  void _init();
  void _print_assignments();

  

  std::vector<Clause> _clauses; 
  std::vector<Status> _assignments;
  
  // output file stream to write to z3py
  std::ofstream _z3_ofs;


  // mapping: assignments (variable) -> clauses' id
  std::unordered_map<int, std::vector<ClauseSatisfiability>> _var_to_clauses;

  // counter for currently satisfied clauses
  size_t _num_sat_clauses = 0;

  // lookup for the status of each clause
  std::vector<Status> _clauses_status;
};



}  // end of namespace --------------------------------------------------------





