### 1. Basic Operations (examples/basic/)

#### 1. `identity`
```json
{
  "name": "identity",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": { "Idx": 0 }
      }
    }
  ]
}
```

#### 2. `add_one`
```json
{
  "name": "add_one",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Add",
            { "Idx": 0 },
            { "Lit": { "Int": 1 } }
          ]
        }
      }
    }
  ]
}
```

#### 3. `double`
```json
{
  "name": "double",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Mul",
            { "Idx": 0 },
            { "Lit": { "Int": 2 } }
          ]
        }
      }
    }
  ]
}
```

#### 4. `square`
```json
{
  "name": "square",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Mul",
            { "Idx": 0 },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 5. `max_two`
```json
{
  "name": "max_two",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Geq",
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            },
            "then_": { "Idx": 1 },
            "else_": { "Idx": 0 }
          }
        }
      }
    }
  ]
}
```

#### 6. `min_two`
```json
{
  "name": "min_two",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Leq",
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            },
            "then_": { "Idx": 1 },
            "else_": { "Idx": 0 }
          }
        }
      }
    }
  ]
}
```

#### 7. `abs`
```json
{
  "name": "abs",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Lt",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": {
              "UnaryOp": [
                "Neg",
                { "Idx": 0 }
              ]
            },
            "else_": { "Idx": 0 }
          }
        }
      }
    }
  ]
}
```

#### 8. `sign`
```json
{
  "name": "sign",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Lt",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": { "Lit": { "Int": -1 } },
            "else_": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Gt",
                    { "Idx": 0 },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": { "Lit": { "Int": 1 } },
                "else_": { "Lit": { "Int": 0 } }
              }
            }
          }
        }
      }
    }
  ]
}
```

#### 9. `is_even`
```json
{
  "name": "is_even",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Eq",
            {
              "BinOp": [
                "Mod",
                { "Idx": 0 },
                { "Lit": { "Int": 2 } }
              ]
            },
            { "Lit": { "Int": 0 } }
          ]
        }
      }
    }
  ]
}
```

#### 10. `is_positive`
```json
{
  "name": "is_positive",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Gt",
            { "Idx": 0 },
            { "Lit": { "Int": 0 } }
          ]
        }
      }
    }
  ]
}
```

### 2. Recursion (examples/recursion/)

#### 1. `factorial`
```json
{
  "name": "factorial",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Leq",
                { "Idx": 0 },
                { "Lit": { "Int": 1 } }
              ]
            },
            "then_": { "Lit": { "Int": 1 } },
            "else_": {
              "BinOp": [
                "Mul",
                { "Idx": 0 },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 2. `fibonacci`
```json
{
  "name": "fibonacci",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Leq",
                { "Idx": 0 },
                { "Lit": { "Int": 1 } }
              ]
            },
            "then_": { "Idx": 0 },
            "else_": {
              "BinOp": [
                "Add",
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    }
                  ]
                },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 2 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 3. `sum_to_n`
```json
{
  "name": "sum_to_n",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Lt",
                { "Idx": 0 },
                { "Lit": { "Int": 1 } }
              ]
            },
            "then_": { "Lit": { "Int": 0 } },
            "else_": {
              "BinOp": [
                "Add",
                { "Idx": 0 },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 4. `power`
```json
{
  "name": "power",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": { "Lit": { "Int": 1 } },
            "else_": {
              "BinOp": [
                "Mul",
                { "Idx": 1 },
                {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 1 },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 5. `gcd`
```json
{
  "name": "gcd",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": { "Idx": 1 },
            "else_": {
              "App": [
                { "Name": "main" },
                { "Idx": 0 },
                {
                  "BinOp": [
                    "Mod",
                    { "Idx": 1 },
                    { "Idx": 0 }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 6. `lcm`
```json
{
  "name": "lcm",
  "decls": [
    {
      "Fn": {
        "name": "gcd",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": { "Idx": 1 },
            "else_": {
              "App": [
                { "Name": "gcd" },
                { "Idx": 0 },
                {
                  "BinOp": [
                    "Mod",
                    { "Idx": 1 },
                    { "Idx": 0 }
                  ]
                }
              ]
            }
          }
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Div",
            {
              "BinOp": [
                "Mul",
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            },
            {
              "App": [
                { "Name": "gcd" },
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 7. `ackermann`
```json
{
  "name": "ackermann",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 1 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": {
              "BinOp": [
                "Add",
                { "Idx": 0 },
                { "Lit": { "Int": 1 } }
              ]
            },
            "else_": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Eq",
                    { "Idx": 0 },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 1 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    { "Lit": { "Int": 1 } }
                  ]
                },
                "else_": {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 1 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    {
                      "App": [
                        { "Name": "main" },
                        { "Idx": 1 },
                        {
                          "BinOp": [
                            "Sub",
                            { "Idx": 0 },
                            { "Lit": { "Int": 1 } }
                          ]
                        }
                      ]
                    }
                  ]
                }
              }
            }
          }
        }
      }
    }
  ]
}
```

#### 8. `sudan`
```json
{
  "name": "sudan",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                {
                  "Fn": [
                    { "Prim": "I64" },
                    { "Prim": "I64" }
                  ]
                }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 2 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": {
              "BinOp": [
                "Add",
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            },
            "else_": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Eq",
                    { "Idx": 0 },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": { "Idx": 1 },
                "else_": {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 2 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    {
                      "App": [
                        { "Name": "main" },
                        { "Idx": 2 },
                        { "Idx": 1 },
                        {
                          "BinOp": [
                            "Sub",
                            { "Idx": 0 },
                            { "Lit": { "Int": 1 } }
                          ]
                        }
                      ]
                    },
                    {
                      "BinOp": [
                        "Add",
                        {
                          "App": [
                            { "Name": "main" },
                            { "Idx": 2 },
                            { "Idx": 1 },
                            {
                              "BinOp": [
                                "Sub",
                                { "Idx": 0 },
                                { "Lit": { "Int": 1 } }
                              ]
                            }
                          ]
                        },
                        { "Idx": 0 }
                      ]
                    }
                  ]
                }
              }
            }
          }
        }
      }
    }
  ]
}
```

#### 9. `collatz_len`
```json
{
  "name": "collatz_len",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 0 },
                { "Lit": { "Int": 1 } }
              ]
            },
            "then_": { "Lit": { "Int": 0 } },
            "else_": {
              "BinOp": [
                "Add",
                { "Lit": { "Int": 1 } },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "If": {
                        "cond": {
                          "BinOp": [
                            "Eq",
                            {
                              "BinOp": [
                                "Mod",
                                { "Idx": 0 },
                                { "Lit": { "Int": 2 } }
                              ]
                            },
                            { "Lit": { "Int": 0 } }
                          ]
                        },
                        "then_": {
                          "BinOp": [
                            "Div",
                            { "Idx": 0 },
                            { "Lit": { "Int": 2 } }
                          ]
                        },
                        "else_": {
                          "BinOp": [
                            "Add",
                            {
                              "BinOp": [
                                "Mul",
                                { "Lit": { "Int": 3 } },
                                { "Idx": 0 }
                              ]
                            },
                            { "Lit": { "Int": 1 } }
                          ]
                        }
                      }
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 10. `digit_sum`
```json
{
  "name": "digit_sum",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Lt",
                { "Idx": 0 },
                { "Lit": { "Int": 10 } }
              ]
            },
            "then_": { "Idx": 0 },
            "else_": {
              "BinOp": [
                "Add",
                {
                  "BinOp": [
                    "Mod",
                    { "Idx": 0 },
                    { "Lit": { "Int": 10 } }
                  ]
                },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Div",
                        { "Idx": 0 },
                        { "Lit": { "Int": 10 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 11. `reverse_num`
```json
{
  "name": "reverse_num",
  "decls": [
    {
      "Fn": {
        "name": "rev",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 1 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": { "Idx": 0 },
            "else_": {
              "App": [
                { "Name": "rev" },
                {
                  "BinOp": [
                    "Div",
                    { "Idx": 1 },
                    { "Lit": { "Int": 10 } }
                  ]
                },
                {
                  "BinOp": [
                    "Add",
                    {
                      "BinOp": [
                        "Mul",
                        { "Idx": 0 },
                        { "Lit": { "Int": 10 } }
                      ]
                    },
                    {
                      "BinOp": [
                        "Mod",
                        { "Idx": 1 },
                        { "Lit": { "Int": 10 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "rev" },
            { "Idx": 0 },
            { "Lit": { "Int": 0 } }
          ]
        }
      }
    }
  ]
}
```

#### 12. `hyperop`
```json
{
  "name": "hyperop",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                {
                  "Fn": [
                    { "Prim": "I64" },
                    { "Prim": "I64" }
                  ]
                }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 2 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Eq",
                    { "Idx": 0 },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": {
                  "BinOp": [
                    "Add",
                    { "Idx": 1 },
                    { "Lit": { "Int": 1 } }
                  ]
                },
                "else_": {
                  "If": {
                    "cond": {
                      "BinOp": [
                        "Eq",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    "then_": {
                      "BinOp": [
                        "Add",
                        { "Idx": 1 },
                        { "Idx": 1 }
                      ]
                    },
                    "else_": {
                      "App": [
                        { "Name": "main" },
                        {
                          "BinOp": [
                            "Sub",
                            { "Idx": 2 },
                            { "Lit": { "Int": 1 } }
                          ]
                        },
                        { "Idx": 1 },
                        {
                          "App": [
                            { "Name": "main" },
                            { "Idx": 2 },
                            { "Idx": 1 },
                            {
                              "BinOp": [
                                "Sub",
                                { "Idx": 1 },
                                { "Lit": { "Int": 1 } }
                              ]
                            }
                          ]
                        }
                      ]
                    }
                  }
                }
              }
            },
            "else_": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Eq",
                    { "Idx": 0 },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": { "Idx": 1 },
                "else_": {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 2 },
                    {
                      "App": [
                        { "Name": "main" },
                        { "Idx": 2 },
                        { "Idx": 1 },
                        {
                          "BinOp": [
                            "Sub",
                            { "Idx": 0 },
                            { "Lit": { "Int": 1 } }
                          ]
                        }
                      ]
                    },
                    { "Idx": 1 }
                  ]
                }
              }
            }
          }
        }
      }
    }
  ]
}
```

#### 13. `tak`
```json
{
  "name": "tak",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                {
                  "Fn": [
                    { "Prim": "I64" },
                    { "Prim": "I64" }
                  ]
                }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Leq",
                { "Idx": 2 },
                { "Idx": 1 }
              ]
            },
            "then_": { "Idx": 0 },
            "else_": {
              "App": [
                { "Name": "main" },
                {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 2 },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 1 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    { "Idx": 0 }
                  ]
                },
                {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 1 },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    { "Idx": 2 }
                  ]
                },
                {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 0 },
                    {
                      "BinOp": [
                        "Sub",
                        { "Idx": 2 },
                        { "Lit": { "Int": 1 } }
                      ]
                    },
                    { "Idx": 1 }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

#### 14. `mccarthy91`
```json
{
  "name": "mccarthy91",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Gt",
                { "Idx": 0 },
                { "Lit": { "Int": 100 } }
              ]
            },
            "then_": {
              "BinOp": [
                "Sub",
                { "Idx": 0 },
                { "Lit": { "Int": 10 } }
              ]
            },
            "else_": {
              "App": [
                { "Name": "main" },
                {
                  "App": [
                    { "Name": "main" },
                    {
                      "BinOp": [
                        "Add",
                        { "Idx": 0 },
                        { "Lit": { "Int": 11 } }
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

### 3. Higher-Order Functions (examples/higher-order/)

#### 1. `map_double`
```json
{
  "name": "map_double",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Tensor": [{ "Prim": "I64" }, ["?"]] }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Map",
            {
              "BinOp": [
                "Map",
                {
                  "App": [
                    { "Name": "iota" },
                    { "Idx": 0 }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Add",
                      { "Idx": 0 },
                      { "Lit": { "Int": 1 } }
                    ]
                  }
                }
              ]
            },
            {
              "Lam": {
                "BinOp": [
                  "Mul",
                  { "Idx": 0 },
                  { "Lit": { "Int": 2 } }
                ]
              }
            }
          ]
        }
      }
    }
  ]
}
```

#### 2. `filter_positive`
```json
{
  "name": "filter_positive",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Tensor": [{ "Prim": "I64" }, ["?"]] }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Map",
            {
              "App": [
                { "Name": "iota" },
                { "Idx": 0 }
              ]
            },
            {
              "Lam": {
                "BinOp": [
                  "Add",
                  { "Idx": 0 },
                  { "Lit": { "Int": 1 } }
                ]
              }
            }
          ]
        }
      }
    }
  ]
}
```

#### 3. `fold_sum`
```json
{
  "name": "fold_sum",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Sum",
            {
              "BinOp": [
                "Map",
                {
                  "App": [
                    { "Name": "iota" },
                    { "Idx": 0 }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Add",
                      { "Idx": 0 },
                      { "Lit": { "Int": 1 } }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 4. `fold_product`
```json
{
  "name": "fold_product",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Prod",
            {
              "BinOp": [
                "Map",
                {
                  "App": [
                    { "Name": "iota" },
                    { "Idx": 0 }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Add",
                      { "Idx": 0 },
                      { "Lit": { "Int": 1 } }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 5. `compose`
```json
{
  "name": "compose",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            {
              "BinOp": [
                "Compose",
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Idx": 0 }
                    ]
                  }
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Lit": { "Int": 2 } }
                    ]
                  }
                }
              ]
            },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 6. `apply_twice`
```json
{
  "name": "apply_twice",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            {
              "BinOp": [
                "Compose",
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Lit": { "Int": 2 } }
                    ]
                  }
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Lit": { "Int": 2 } }
                    ]
                  }
                }
              ]
            },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 7. `all_positive`
```json
{
  "name": "all_positive",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Eq",
            {
              "App": [
                { "Name": "len" },
                {
                  "BinOp": [
                    "Filter",
                    {
                      "BinOp": [
                        "Map",
                        {
                          "App": [
                            { "Name": "iota" },
                            { "Idx": 0 }
                          ]
                        },
                        {
                          "Lam": {
                            "BinOp": [
                              "Add",
                              { "Idx": 0 },
                              { "Lit": { "Int": 1 } }
                            ]
                          }
                        }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Gt",
                          { "Idx": 0 },
                          { "Lit": { "Int": 0 } }
                        ]
                      }
                    }
                  ]
                }
              ]
            },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 8. `any_negative`
```json
{
  "name": "any_negative",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Gt",
            {
              "App": [
                { "Name": "len" },
                {
                  "BinOp": [
                    "Filter",
                    {
                      "BinOp": [
                        "Map",
                        {
                          "App": [
                            { "Name": "iota" },
                            {
                              "BinOp": [
                                "Add",
                                {
                                  "BinOp": [
                                    "Mul",
                                    { "Idx": 0 },
                                    { "Lit": { "Int": 2 } }
                                  ]
                                },
                                { "Lit": { "Int": 1 } }
                              ]
                            }
                          ]
                        },
                        {
                          "Lam": {
                            "BinOp": [
                              "Sub",
                              { "Idx": 0 },
                              { "Idx": 0 }
                            ]
                          }
                        }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Lt",
                          { "Idx": 0 },
                          { "Lit": { "Int": 0 } }
                        ]
                      }
                    }
                  ]
                }
              ]
            },
            { "Lit": { "Int": 0 } }
          ]
        }
      }
    }
  ]
}
```

#### 9. `count_if`
```json
{
  "name": "count_if",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "len" },
            {
              "BinOp": [
                "Filter",
                {
                  "BinOp": [
                    "Map",
                    {
                      "App": [
                        { "Name": "iota" },
                        { "Idx": 0 }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Add",
                          { "Idx": 0 },
                          { "Lit": { "Int": 1 } }
                        ]
                      }
                    }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Eq",
                      {
                        "BinOp": [
                          "Mod",
                          { "Idx": 0 },
                          { "Lit": { "Int": 2 } }
                        ]
                      },
                      { "Lit": { "Int": 0 } }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 10. `pipeline`
```json
{
  "name": "pipeline",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Sum",
            {
              "BinOp": [
                "Map",
                {
                  "BinOp": [
                    "Filter",
                    {
                      "BinOp": [
                        "Map",
                        {
                          "App": [
                            { "Name": "iota" },
                            { "Idx": 0 }
                          ]
                        },
                        {
                          "Lam": {
                            "BinOp": [
                              "Add",
                              { "Idx": 0 },
                              { "Lit": { "Int": 1 } }
                            ]
                          }
                        }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Eq",
                          {
                            "BinOp": [
                              "Mod",
                              { "Idx": 0 },
                              { "Lit": { "Int": 2 } }
                            ]
                          },
                          { "Lit": { "Int": 0 } }
                        ]
                      }
                    }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Idx": 0 }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

### 4. Numeric Algorithms (examples/numeric/)

#### 1. `gamma_fact`
```json
{
  "name": "gamma_fact",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "F64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "gamma" },
            {
              "BinOp": [
                "Add",
                { "Idx": 0 },
                { "Lit": { "Float": 1.0 } }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 2. `gamma_half`
```json
{
  "name": "gamma_half",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "F64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "gamma" },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 3. `sum_squares`
```json
{
  "name": "sum_squares",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Sum",
            {
              "BinOp": [
                "Map",
                {
                  "BinOp": [
                    "Map",
                    {
                      "App": [
                        { "Name": "iota" },
                        { "Idx": 0 }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Add",
                          { "Idx": 0 },
                          { "Lit": { "Int": 1 } }
                        ]
                      }
                    }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Mul",
                      { "Idx": 0 },
                      { "Idx": 0 }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 4. `product_range`
```json
{
  "name": "product_range",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Prod",
            {
              "BinOp": [
                "Map",
                {
                  "App": [
                    { "Name": "iota" },
                    { "Idx": 0 }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Add",
                      { "Idx": 0 },
                      { "Lit": { "Int": 1 } }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 5. `harmonic`
```json
{
  "name": "harmonic",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Sum",
            {
              "BinOp": [
                "Map",
                {
                  "BinOp": [
                    "Map",
                    {
                      "App": [
                        { "Name": "iota" },
                        { "Idx": 0 }
                      ]
                    },
                    {
                      "Lam": {
                        "BinOp": [
                          "Add",
                          { "Idx": 0 },
                          { "Lit": { "Int": 1 } }
                        ]
                      }
                    }
                  ]
                },
                {
                  "Lam": {
                    "BinOp": [
                      "Div",
                      { "Lit": { "Float": 1.0 } },
                      {
                        "App": [
                          { "Name": "float" },
                          { "Idx": 0 }
                        ]
                      }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 6. `exp_taylor`
```json
{
  "name": "exp_taylor",
  "decls": [
    {
      "Fn": {
        "name": "taylor_term",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "F64" },
                { "Prim": "F64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Div",
            {
              "App": [
                { "Name": "pow" },
                { "Idx": 1 },
                {
                  "App": [
                    { "Name": "float" },
                    { "Idx": 0 }
                  ]
                }
              ]
            },
            {
              "App": [
                { "Name": "gamma" },
                {
                  "BinOp": [
                    "Add",
                    {
                      "App": [
                        { "Name": "float" },
                        { "Idx": 0 }
                      ]
                    },
                    { "Lit": { "Float": 1.0 } }
                  ]
                }
              ]
            }
          ]
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "F64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Sum",
            {
              "BinOp": [
                "Map",
                {
                  "App": [
                    { "Name": "iota" },
                    { "Lit": { "Int": 20 } }
                  ]
                },
                {
                  "Lam": {
                    "App": [
                      { "Name": "taylor_term" },
                      { "Idx": 0 },
                      { "Idx": 1 }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 7. `pi_leibniz`
```json
{
  "name": "pi_leibniz",
  "decls": [
    {
      "Fn": {
        "name": "leibniz_term",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Div",
            {
              "If": {
                "cond": {
                  "BinOp": [
                    "Eq",
                    {
                      "BinOp": [
                        "Mod",
                        { "Idx": 0 },
                        { "Lit": { "Int": 2 } }
                      ]
                    },
                    { "Lit": { "Int": 0 } }
                  ]
                },
                "then_": { "Lit": { "Float": 1.0 } },
                "else_": { "Lit": { "Float": -1.0 } }
              }
            },
            {
              "BinOp": [
                "Add",
                {
                  "BinOp": [
                    "Mul",
                    { "Lit": { "Float": 2.0 } },
                    {
                      "App": [
                        { "Name": "float" },
                        { "Idx": 0 }
                      ]
                    }
                  ]
                },
                { "Lit": { "Float": 1.0 } }
              ]
            }
          ]
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Mul",
            { "Lit": { "Float": 4.0 } },
            {
              "UnaryOp": [
                "Sum",
                {
                  "BinOp": [
                    "Map",
                    {
                      "App": [
                        { "Name": "iota" },
                        { "Idx": 0 }
                      ]
                    },
                    {
                      "Lam": {
                        "App": [
                          { "Name": "leibniz_term" },
                          { "Idx": 0 }
                        ]
                      }
                    }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 8. `sqrt_newton`
```json
{
  "name": "sqrt_newton",
  "decls": [
    {
      "Fn": {
        "name": "newton_step",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "F64" },
            {
              "Fn": [
                { "Prim": "F64" },
                { "Prim": "F64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "BinOp": [
            "Sub",
            { "Idx": 0 },
            {
              "BinOp": [
                "Div",
                {
                  "BinOp": [
                    "Sub",
                    {
                      "BinOp": [
                        "Mul",
                        { "Idx": 0 },
                        { "Idx": 0 }
                      ]
                    },
                    { "Idx": 1 }
                  ]
                },
                {
                  "BinOp": [
                    "Mul",
                    { "Lit": { "Float": 2.0 } },
                    { "Idx": 0 }
                  ]
                }
              ]
            }
          ]
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "F64" },
            { "Prim": "F64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "Let": {
            "pattern": { "Var": "x0" },
            "type_": null,
            "value": {
              "BinOp": [
                "Div",
                { "Idx": 0 },
                { "Lit": { "Float": 2.0 } }
              ]
            },
            "body": {
              "App": [
                { "Name": "foldl" },
                { "Name": "newton_step" },
                { "Idx": 0 },
                {
                  "App": [
                    { "Name": "iota" },
                    { "Lit": { "Int": 10 } }
                  ]
                }
              ]
            }
          }
        }
      }
    }
  ]
}
```

### 5. Algorithms (examples/algorithms/)

#### 1. `binary_search`
```json
{
  "name": "binary_search",
  "decls": [
    {
      "Fn": {
        "name": "bs",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                {
                  "Fn": [
                    { "Prim": "I64" },
                    { "Prim": "I64" }
                  ]
                }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Gt",
                { "Idx": 1 },
                { "Idx": 0 }
              ]
            },
            "then_": { "Lit": { "Int": -1 } },
            "else_": {
              "Let": {
                "pattern": { "Var": "mid" },
                "type_": null,
                "value": {
                  "BinOp": [
                    "Div",
                    {
                      "BinOp": [
                        "Add",
                        { "Idx": 1 },
                        { "Idx": 0 }
                      ]
                    },
                    { "Lit": { "Int": 2 } }
                  ]
                },
                "body": {
                  "If": {
                    "cond": {
                      "BinOp": [
                        "Eq",
                        { "Idx": 0 },
                        { "Idx": 2 }
                      ]
                    },
                    "then_": { "Idx": 0 },
                    "else_": {
                      "If": {
                        "cond": {
                          "BinOp": [
                            "Lt",
                            { "Idx": 0 },
                            { "Idx": 2 }
                          ]
                        },
                        "then_": {
                          "App": [
                            { "Name": "bs" },
                            {
                              "BinOp": [
                                "Add",
                                { "Idx": 0 },
                                { "Lit": { "Int": 1 } }
                              ]
                            },
                            { "Idx": 1 },
                            { "Idx": 2 }
                          ]
                        },
                        "else_": {
                          "App": [
                            { "Name": "bs" },
                            { "Idx": 0 },
                            {
                              "BinOp": [
                                "Sub",
                                { "Idx": 1 },
                                { "Lit": { "Int": 1 } }
                              ]
                            },
                            { "Idx": 2 }
                          ]
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                { "Prim": "I64" }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "bs" },
            { "Lit": { "Int": 1 } },
            { "Idx": 1 },
            { "Idx": 0 }
          ]
        }
      }
    }
  ]
}
```

#### 2. `isPrime`
```json
{
  "name": "isPrime",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "Let": {
            "pattern": { "Var": "sqrt_n" },
            "type_": null,
            "value": {
              "UnaryOp": [
                "Floor",
                {
                  "UnaryOp": [
                    "Sqrt",
                    {
                      "App": [
                        { "Name": "float" },
                        { "Idx": 0 }
                      ]
                    }
                  ]
                }
              ]
            },
            "body": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Lt",
                    { "Idx": 1 },
                    { "Lit": { "Int": 2 } }
                  ]
                },
                "then_": { "Lit": "False" },
                "else_": {
                  "BinOp": [
                    "Eq",
                    {
                      "App": [
                        { "Name": "len" },
                        {
                          "BinOp": [
                            "Filter",
                            {
                              "App": [
                                { "Name": "range" },
                                { "Lit": { "Int": 2 } },
                                {
                                  "BinOp": [
                                    "Add",
                                    { "Idx": 0 },
                                    { "Lit": { "Int": 1 } }
                                  ]
                                }
                              ]
                            },
                            {
                              "Lam": {
                                "BinOp": [
                                  "Eq",
                                  {
                                    "BinOp": [
                                      "Mod",
                                      { "Idx": 1 },
                                      { "Idx": 0 }
                                    ]
                                  },
                                  { "Lit": { "Int": 0 } }
                                ]
                              }
                            }
                          ]
                        }
                      ]
                    },
                    { "Lit": { "Int": 0 } }
                  ]
                }
              }
            }
          }
        }
      }
    }
  ]
}
```

#### 3. `count_primes`
```json
{
  "name": "count_primes",
  "decls": [
    {
      "Fn": {
        "name": "is_prime",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "Bool" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "Let": {
            "pattern": { "Var": "sqrt_n" },
            "type_": null,
            "value": {
              "UnaryOp": [
                "Floor",
                {
                  "UnaryOp": [
                    "Sqrt",
                    {
                      "App": [
                        { "Name": "float" },
                        { "Idx": 0 }
                      ]
                    }
                  ]
                }
              ]
            },
            "body": {
              "If": {
                "cond": {
                  "BinOp": [
                    "Lt",
                    { "Idx": 1 },
                    { "Lit": { "Int": 2 } }
                  ]
                },
                "then_": { "Lit": "False" },
                "else_": {
                  "BinOp": [
                    "Eq",
                    {
                      "App": [
                        { "Name": "len" },
                        {
                          "BinOp": [
                            "Filter",
                            {
                              "App": [
                                { "Name": "range" },
                                { "Lit": { "Int": 2 } },
                                {
                                  "BinOp": [
                                    "Add",
                                    { "Idx": 0 },
                                    { "Lit": { "Int": 1 } }
                                  ]
                                }
                              ]
                            },
                            {
                              "Lam": {
                                "BinOp": [
                                  "Eq",
                                  {
                                    "BinOp": [
                                      "Mod",
                                      { "Idx": 1 },
                                      { "Idx": 0 }
                                    ]
                                  },
                                  { "Lit": { "Int": 0 } }
                                ]
                              }
                            }
                          ]
                        }
                      ]
                    },
                    { "Lit": { "Int": 0 } }
                  ]
                }
              }
            }
          }
        }
      }
    },
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "App": [
            { "Name": "len" },
            {
              "BinOp": [
                "Filter",
                {
                  "App": [
                    { "Name": "range" },
                    { "Lit": { "Int": 2 } },
                    {
                      "BinOp": [
                        "Add",
                        { "Idx": 0 },
                        { "Lit": { "Int": 1 } }
                      ]
                    }
                  ]
                },
                {
                  "Lam": {
                    "App": [
                      { "Name": "is_prime" },
                      { "Idx": 0 }
                    ]
                  }
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 4. `nth_prime`
```json
{
  "name": "nth_prime",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "Let": {
            "pattern": { "Var": "find_nth" },
            "type_": null,
            "value": {
              "Lam": {
                "Lam": {
                  "Lam": {
                    "If": {
                      "cond": {
                        "BinOp": [
                          "Eq",
                          { "Idx": 1 },
                          { "Idx": 2 }
                        ]
                      },
                      "then_": { "Idx": 0 },
                      "else_": {
                        "App": [
                          { "Idx": 3 },
                          { "Idx": 2 },
                          {
                            "BinOp": [
                              "Add",
                              { "Idx": 1 },
                              {
                                "If": {
                                  "cond": {
                                    "App": [
                                      { "Name": "is_prime" },
                                      {
                                        "BinOp": [
                                          "Add",
                                          { "Idx": 0 },
                                          { "Lit": { "Int": 1 } }
                                        ]
                                      }
                                    ]
                                  },
                                  "then_": { "Lit": { "Int": 1 } },
                                  "else_": { "Lit": { "Int": 0 } }
                                }
                              }
                            ]
                          },
                          {
                            "BinOp": [
                              "Add",
                              { "Idx": 0 },
                              { "Lit": { "Int": 1 } }
                            ]
                          }
                        ]
                      }
                    }
                  }
                }
              }
            },
            "body": {
              "App": [
                { "Idx": 0 },
                { "Idx": 1 },
                { "Lit": { "Int": 0 } },
                { "Lit": { "Int": 2 } }
              ]
            }
          }
        }
      }
    }
  ]
}
```

(Note: Assumes "is_prime" defined as in previous.)

#### 5. `isqrt`
```json
{
  "name": "isqrt",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            { "Prim": "I64" }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "UnaryOp": [
            "Floor",
            {
              "UnaryOp": [
                "Sqrt",
                {
                  "App": [
                    { "Name": "float" },
                    { "Idx": 0 }
                  ]
                }
              ]
            }
          ]
        }
      }
    }
  ]
}
```

#### 6. `modpow`
```json
{
  "name": "modpow",
  "decls": [
    {
      "Fn": {
        "name": "main",
        "type_params": [],
        "signature": {
          "Fn": [
            { "Prim": "I64" },
            {
              "Fn": [
                { "Prim": "I64" },
                {
                  "Fn": [
                    { "Prim": "I64" },
                    { "Prim": "I64" }
                  ]
                }
              ]
            }
          ]
        },
        "effects": [],
        "constraints": [],
        "preconditions": [],
        "postconditions": [],
        "body": {
          "If": {
            "cond": {
              "BinOp": [
                "Eq",
                { "Idx": 0 },
                { "Lit": { "Int": 0 } }
              ]
            },
            "then_": {
              "BinOp": [
                "Mod",
                { "Lit": { "Int": 1 } },
                { "Idx": 2 }
              ]
            },
            "else_": {
              "Let": {
                "pattern": { "Var": "half" },
                "type_": null,
                "value": {
                  "App": [
                    { "Name": "main" },
                    { "Idx": 1 },
                    {
                      "BinOp": [
                        "Div",
                        { "Idx": 0 },
                        { "Lit": { "Int": 2 } }
                      ]
                    },
                    { "Idx": 2 }
                  ]
                },
                "body": {
                  "If": {
                    "cond": {
                      "BinOp": [
                        "Eq",
                        {
                          "BinOp": [
                            "Mod",
                            { "Idx": 2 },
                            { "Lit": { "Int": 2 } }
                          ]
                        },
                        { "Lit": { "Int": 0 } }
                      ]
                    },
                    "then_": {
                      "BinOp": [
                        "Mod",
                        {
                          "BinOp": [
                            "Mul",
                            { "Idx": 0 },
                            { "Idx": 0 }
                          ]
                        },
                        { "Idx": 3 }
                      ]
                    },
                    "else_": {
                      "BinOp": [
                        "Mod",
                        {
                          "BinOp": [
                            "Mul",
                            {
                              "BinOp": [
                                "Mod",
                                {
                                  "BinOp": [
                                    "Mul",
                                    { "Idx": 0 },
                                    { "Idx": 0 }
                                  ]
                                },
                                { "Idx": 3 }
                              ]
                            },
                            { "Idx": 1 }
                          ]
                        },
                        { "Idx": 3 }
                      ]
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  ]
}
```
