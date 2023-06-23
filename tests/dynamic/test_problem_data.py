from io import StringIO

from optlis.dynamic.problem_data import _write_instance


def test_export_instance(example_dynamic_instance):
    """Tests the function to export a problem instance to a text file."""
    inst = example_dynamic_instance
    outfile = StringIO()
    _write_instance(inst, outfile)
    outfile.seek(0)
    assert outfile.read() == "\n".join(
        (
            "# format: dynamic",
            "3",
            "0 0.00",
            "1 0.50",
            "2 0.80",
            "0 0.00",
            "1 0.00",
            "2 0.01",
            "0 0.00 0.00 0.00",
            "1 0.00 0.00 0.05",
            "2 0.00 0.00 0.00",
            "9",
            "0 0 1 1 0",
            "1 1 0 0 5",
            "2 1 0 0 5",
            "3 1 0 0 5",
            "4 1 0 0 5",
            "5 1 0 0 5",
            "6 1 0 0 5",
            "7 1 0 0 5",
            "8 1 0 0 5",
            "9",
            "0 0.00 0.00 0.00",
            "1 0.00 1.00 0.50",
            "2 0.00 1.00 0.50",
            "3 0.00 1.00 0.50",
            "4 0.00 1.00 0.50",
            "5 0.00 1.00 0.50",
            "6 0.00 1.00 0.50",
            "7 0.00 1.00 0.50",
            "8 0.00 1.00 0.50",
            "106\n",
        )
    )
